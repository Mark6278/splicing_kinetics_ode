import os

TMP = os.environ.get("TMPDIR", "/tmp")
import json, os

IS_HPC = str(config.get("mode", "laptop")).lower() == "hpc"
PLATFORM = str(config.get("platform", "short")).lower()
LONG_PRESET = str(config.get("long_preset", "splice:hq"))

CTRL = [s.strip() for s in str(config.get("control_samples", config.get("samples", "SRR4421449"))).split(",") if s.strip()]
TREAT = [s.strip() for s in str(config.get("treatment_samples", "")).split(",") if s.strip()]
SAMPLES = list(dict.fromkeys(CTRL + TREAT))
GENES = [g.strip() for g in str(config.get("genes", "SRSF3")).split(",")]
GENE_TAG = "_".join(GENES)
PAIRED = str(config.get("paired", "true")).lower() == "true"

BUILD, CHROM = str(config.get("build", "hg38")), str(config.get("chrom", "all" if IS_HPC else "17"))
REF = f"{BUILD}_full" if CHROM == "all" else f"{BUILD}_chr{CHROM}"
THREADS = int(config.get("threads", 16 if IS_HPC else 2))
STAR_GB = float(config.get("star_mem_gb", 36.0 if IS_HPC else 1.8))
MAX_READS = int(config.get("max_reads", 0))
LOCUS_FILTER = str(config.get("locus_filter", "false")).lower() == "true"
FAST_ALIGN = str(config.get("fast_align", "false" if IS_HPC else "true")).lower() == "true"
SPARSE = int(config.get("sparse_index", 1 if IS_HPC else 2))
NBASES = int(config.get("sa_nbases", 14 if CHROM == "all" else 12))
MATES = ["1", "2"] if (PLATFORM == "short" and PAIRED) else ["1"]

FASTA, GTF = config.get("fasta", f"refs/{REF}.fa"), config.get("gtf", f"refs/{REF}.annotation.gtf")
IDX = config.get("index", f"refs/{REF}_STAR_index")
LOCUS_FA, LOCUS_REG = f"refs/{REF}_{GENE_TAG}_locus.fa", f"refs/{REF}_{GENE_TAG}_regions.txt"
TARGET_DIR, RESULTS_DIR = f"target_data/{REF}_{PLATFORM}", f"results/{REF}_{PLATFORM}"


def get_rlen(jsons):
    if "read_length" in config:
        return int(config["read_length"])
    for p in jsons:
        if os.path.exists(p):
            r = json.load(open(p)).get("summary", {}).get("after_filtering", {}).get("read1_mean_length")
            if r:
                return max(30, int(round(float(r))))
    return 100


rule all:
    input:
        expand(f"{RESULTS_DIR}/{{gene}}_splicing_dynamics.png", gene=GENES),
        expand(f"{RESULTS_DIR}/{{gene}}_sf_inference.csv", gene=GENES)


rule fetch_reference:
    output: fa=protected(FASTA), gtf=protected(GTF)
    resources: mem_mb=8000 if IS_HPC else 2000, runtime=120
    run:
        import gzip
        os.makedirs(os.path.dirname(output.fa) or "refs", exist_ok=True)
        GENCODE = {
            "hg38": "rsync://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz",
            "hg19": "rsync://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz",
            "mm39": "rsync://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M35/gencode.vM35.annotation.gtf.gz",
            "mm10": "rsync://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
        }
        gtf_url = GENCODE.get(BUILD, f"rsync://hgdownload.soe.ucsc.edu/goldenPath/{BUILD}/bigZips/genes/{BUILD}.ncbiRefSeq.gtf.gz")
        t_fa, t_gtf = f"refs/tmp_{REF}.fa.gz", f"refs/tmp_{REF}.gtf.gz"
        if CHROM != "all" and BUILD in GENCODE:
            shell(f"rsync -az rsync://hgdownload.soe.ucsc.edu/goldenPath/{BUILD}/chromosomes/chr{CHROM}.fa.gz {t_fa} && gunzip -c {t_fa} > {{output.fa}} && rm -f {t_fa}")
        else:
            shell(f"rsync -az rsync://hgdownload.soe.ucsc.edu/goldenPath/{BUILD}/bigZips/{BUILD}.fa.gz {t_fa}")
            keep = CHROM == "all"
            with gzip.open(t_fa, "rt") as fin, open(output.fa, "w") as fout:
                for line in fin:
                    if line.startswith(">"):
                        keep = (CHROM == "all") or (line.strip().split()[0] == f">chr{CHROM}")
                    if keep:
                        fout.write(line)
            os.remove(t_fa)
        shell(f"rsync -az {gtf_url} {t_gtf}")
        with gzip.open(t_gtf, "rt") as fin, open(output.gtf, "w") as fout:
            for line in fin:
                if CHROM == "all" or line.startswith("#") or line.startswith(f"chr{CHROM}\t"):
                    fout.write(line)
        os.remove(t_gtf)


rule extract_target_locus:
    input: fa=FASTA, gtf=GTF
    output: locus_fa=LOCUS_FA, regions_txt=LOCUS_REG
    resources: mem_mb=2000, runtime=15
    run:
        os.makedirs("refs", exist_ok=True)
        targets = {g.upper() for g in GENES}
        regs = {f"{p[0]}:{max(1,int(p[3])-5000)}-{int(p[4])+5000}" for line in open(input.gtf) if not line.startswith("#") and len(p := line.rstrip("\n").split("\t")) >= 9 and p[2] == "gene" and any(f'GENE_NAME "{g}"' in p[8].upper() or f'GENE_ID "{g}' in p[8].upper() for g in targets)}
        if not regs:
            raise ValueError(f"None of {GENES} found in {input.gtf}")
        open(output.regions_txt, "w").write("\n".join(sorted(regs)) + "\n")
        shell("samtools faidx {input.fa} && xargs -a {output.regions_txt} samtools faidx {input.fa} > {output.locus_fa}")


rule build_star_index:
    input: fa=FASTA, gtf=GTF
    output: protected(directory(IDX))
    threads: THREADS
    resources: mem_mb=int(STAR_GB * 1200), runtime=240
    shell: f"mkdir -p {{output}} && rm -rf {TMP}/star_idx_{REF} && STAR --runThreadN {{threads}} --runMode genomeGenerate --genomeDir {{output}} --genomeFastaFiles {{input.fa}} --sjdbGTFfile {{input.gtf}} --genomeSAindexNbases {NBASES} --genomeSAsparseD {SPARSE} --limitGenomeGenerateRAM {int(STAR_GB*1e9)} --outTmpDir {TMP}/star_idx_{REF}"
        

rule bbduk_filter:
    input: reads=expand("data/{{sample}}_{m}.fastq.gz", m=MATES), locus_fa=LOCUS_FA
    output: reads=expand(f"work/filtered/{{{{sample}}}}_{REF}_{GENE_TAG}_{{m}}.fastq.gz", m=MATES)
    threads: THREADS
    resources: mem_mb=4000, runtime=30
    run:
        os.makedirs("work/filtered", exist_ok=True)
        if PAIRED:
            shell("bbduk.sh in1=<(gzip -dc {input.reads[0]} 2>/dev/null || true) in2=<(gzip -dc {input.reads[1]} 2>/dev/null || true) int=f outm1={output.reads[0]} outm2={output.reads[1]} ref={input.locus_fa} k=25 hdist=1 qin=33 ignorebadquality tossjunk -da threads={threads}")
        else:
            shell("bbduk.sh in={input.reads[0]} outm={output.reads[0]} ref={input.locus_fa} k=25 hdist=1 qin=33 ignorebadquality tossjunk -da threads={threads}")


rule fastp_qc:
    input: reads=expand(f"work/filtered/{{{{sample}}}}_{REF}_{GENE_TAG}_{{m}}.fastq.gz" if LOCUS_FILTER else "data/{{sample}}_{m}.fastq.gz", m=MATES)
    output: reads=expand("work/trimmed/{{sample}}_{m}.clean.fastq.gz", m=MATES), html="work/qc/{sample}_fastp.html", json="work/qc/{sample}_fastp.json"
    threads: min(THREADS, 8)
    resources: mem_mb=8000 if IS_HPC else 2000, runtime=60
    run:
        lim = f"--reads_to_process {MAX_READS}" if MAX_READS > 0 else ""
        io = f"-i {input.reads[0]} -I {input.reads[1]} -o {output.reads[0]} -O {output.reads[1]}" if PAIRED else f"-i {input.reads[0]} -o {output.reads[0]}"
        shell(f"fastp {io} -h {{output.html}} -j {{output.json}} --thread {{threads}} {lim}")


rule star_align:
    input: reads=expand("work/trimmed/{{sample}}_{m}.clean.fastq.gz", m=MATES), idx=IDX
    output: bam="work/star/{sample}/Aligned.sortedByCoord.out.bam"
    threads: THREADS
    resources: mem_mb=int(STAR_GB * 1150), runtime=180
    run:
        spd = "--outSAMunmapped None --seedPerWindowNmax 20 --winAnchorMultimapNmax 20" if FAST_ALIGN else ""
        shell(f"rm -rf {TMP}/star_{{wildcards.sample}} && STAR --runThreadN {{threads}} --genomeDir {{input.idx}} --readFilesIn {' '.join(input.reads)} --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outSAMstrandField intronMotif --limitBAMsortRAM {int(min(STAR_GB,32.0)*1e9)} {spd} --outTmpDir {TMP}/star_{{wildcards.sample}} --outFileNamePrefix work/star/{{wildcards.sample}}/")
 

rule rmats_quant:
    input: b1=expand("work/star/{s}/Aligned.sortedByCoord.out.bam", s=CTRL), b1i=expand("work/star/{s}/Aligned.sortedByCoord.out.bam.bai", s=CTRL), b2=expand("work/star/{s}/Aligned.sortedByCoord.out.bam", s=TREAT), b2i=expand("work/star/{s}/Aligned.sortedByCoord.out.bam.bai", s=TREAT), jsons=expand("work/qc/{s}_fastp.json", s=SAMPLES), gtf=GTF
    output: psi="work/rmats/SE.MATS.JC.txt"
    threads: THREADS
    resources: mem_mb=32000 if IS_HPC else 3000, runtime=240
    run:
        os.makedirs("work/rmats", exist_ok=True)
        open("work/rmats/b1.txt", "w").write(",".join(input.b1) + "\n")
        b2_flag = "--b2 work/rmats/b2.txt" if TREAT else "--statoff"
        if TREAT:
            open("work/rmats/b2.txt", "w").write(",".join(input.b2) + "\n")
        shell(f"rm -rf {TMP}/rmats_tmp && rmats.py --b1 work/rmats/b1.txt {b2_flag} --gtf {{input.gtf}} -t {'paired' if PAIRED else 'single'} --readLength {get_rlen(input.jsons)} --variable-read-length --allow-clipping --nthread {{threads}} --od work/rmats --tmp {TMP}/rmats_tmp")
                

rule minimap2_align:
    input: fq="data/{sample}.fastq.gz", fa=FASTA
    output: bam="work/minimap2/{sample}/Aligned.sortedByCoord.out.bam"
    threads: THREADS
    resources: mem_mb=24000 if IS_HPC else 3000, runtime=180
    shell: f"mkdir -p work/minimap2/{{wildcards.sample}} && minimap2 -ax {LONG_PRESET} -uf --secondary=no -t {{threads}} {{input.fa}} {{input.fq}} | samtools sort -@ {{threads}} -m {'2G' if IS_HPC else '512M'} -o {{output.bam}}"


rule index_bam:
    input: "{prefix}/Aligned.sortedByCoord.out.bam"
    output: "{prefix}/Aligned.sortedByCoord.out.bam.bai"
    threads: min(THREADS, 4)
    resources: mem_mb=8000 if IS_HPC else 1500, runtime=30
    shell: "samtools index -@ {threads} {input}"


rule isoquant_quant:
    input: bams=expand("work/minimap2/{s}/Aligned.sortedByCoord.out.bam", s=SAMPLES), bais=expand("work/minimap2/{s}/Aligned.sortedByCoord.out.bam.bai", s=SAMPLES), fa=FASTA, gtf=GTF
    output: tpm="work/isoquant/OUT/OUT.transcript_tpm.tsv"
    threads: THREADS
    resources: mem_mb=32000 if IS_HPC else 3500, runtime=240
    run:
        import glob, re, pandas as pd
        os.makedirs("work/isoquant/OUT", exist_ok=True)
        if IS_HPC:
            dtype = "pacbio_ccs" if "hq" in LONG_PRESET else "nanopore"
            shell(f"isoquant --reference {{input.fa}} --genedb {{input.gtf}} --complete_genedb --bam {{input.bams}} --data_type {dtype} --threads {{threads}} -o work/isoquant")
        else:
            dfs = {}
            for smp, bam in zip(SAMPLES, input.bams):
                gtf_out = f"work/isoquant/OUT/{smp}.gtf"
                shell(f"stringtie -L -e -p {{threads}} -G {{input.gtf}} -o {gtf_out} {bam}")
                tpms = {}
                with open(gtf_out) as f:
                    for line in f:
                        if "\ttranscript\t" in line:
                            m = re.search(r'transcript_id "([^"]+)".*?TPM "([^"]+)"', line)
                            if m:
                                tpms[m.group(1)] = float(m.group(2))
                dfs[smp] = tpms
            pd.DataFrame(dfs).fillna(0.0).to_csv(output.tpm, sep="\t")


rule suppa_psi:
    input: gtf=GTF, tpm="work/isoquant/OUT/OUT.transcript_tpm.tsv"
    output: psi="work/suppa/long_events.psi"
    resources: mem_mb=8000 if IS_HPC else 2000, runtime=30
    shell: f"mkdir -p work/suppa && suppa.py generateEvents -i {{input.gtf}} -o refs/{REF} -e SE -f ioe && suppa.py psiPerEvent -i refs/{REF}_SE_strict.ioe -e {{input.tpm}} -o work/suppa/long_events"


rule ode_modeling:
    input: psi="work/rmats/SE.MATS.JC.txt" if PLATFORM == "short" else "work/suppa/long_events.psi"
    output: sub=f"{TARGET_DIR}/{{gene}}_splicing_psi.csv", plot=f"{RESULTS_DIR}/{{gene}}_splicing_dynamics.png", table=f"{RESULTS_DIR}/{{gene}}_sf_inference.csv"
    threads: min(THREADS, 8)
    resources: mem_mb=4000 if IS_HPC else 1500, runtime=15
    run:
        ext = f"python src/extract.py --input {{input.psi}} --target {{wildcards.gene}} --output {{output.sub}}" if PLATFORM == "short" else f"python src/extract_suppa.py --psi {{input.psi}} --gtf {GTF} --target {{wildcards.gene}} --output {{output.sub}}"
        shell(f"mkdir -p {TARGET_DIR} {RESULTS_DIR} && {ext} && python src/run_pipeline.py --input {{output.sub}} --outdir {RESULTS_DIR} --samples {','.join(SAMPLES)} --fdr {float(config.get('fdr', 1.0))} --workers {{threads}}")