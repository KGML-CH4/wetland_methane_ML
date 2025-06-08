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
    path "results_${test_index}_${rep}.txt"

    script:
    """
    python ${params.repo}/train_baselineML.py \
        ${workdir} \
        ${test_index} \
        ${rep} \
        > results_${test_index}_${rep}.txt
    """
}

workflow {
    ch_a = Channel.of(params.workdir)
    ch_b = Channel.of( 0..params.num_sites )
    ch_c = Channel.of( 0..params.num_reps )
    
    combined_channel = ch_a.combine(ch_b)
    combined_channel = combined_channel.combine(ch_c)
    Train_Model(combined_channel)
}
