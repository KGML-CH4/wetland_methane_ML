params.each { k, v -> 
    println "- $k = $v"
}
println "=============================\n"

process Train_Model {
    publishDir "results/"
    
    tag "${test_index}_${rep}"

    conda "${params.repo}/requirements.yml"

    input:
    tuple val(workdir), val(test_index), val(rep)

    output:
    path "Out/Baseline_ML/results_site_${test_index}_rep_${rep}.txt"

    script:
    """
    python ${params.repo}/train_baselineML.py \
        ${workdir} \
        ${test_index} \
        ${rep}
    """
}

workflow {
//    def outputDir = file("${params.workdir}/Out/LOOCV_baseline_ml")
  //  outputDir.mkdirs()

    ch_a = Channel.of(params.workdir)
    ch_b = Channel.of( 0..(params.num_sites-1) ) //0-index
    ch_c = Channel.of( 0..(params.num_reps-1) )
    
    combined_channel = ch_a.combine(ch_b)
    combined_channel = combined_channel.combine(ch_c)
    Train_Model(combined_channel)
}

