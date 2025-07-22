params.each { k, v -> 
    println "- $k = $v"
}
println "=============================\n"



workflow {
    // train with baseline ML model                       
    ch_a = Channel.of(params.workdir)
    ch_b = Channel.of( 0..(params.num_sites-1) ) //0-index
    ch_c = Channel.of( 0..(params.num_reps-1) )
    combined_channel = ch_a.combine(ch_b)
    combined_channel = combined_channel.combine(ch_c)
    baselineML = Train_baselineML(combined_channel)

    // evaluate
    baselineML
    	.collect() 
    	.map { result_files ->
            tuple("${params.workdir}/Out/Baseline_ML/", "Baseline machine learning", result_files)
    	}
    	.set { inpt }
    Eval(inpt)
}



process Train_baselineML {
    publishDir "${params.workdir}/Out/Baseline_ML/", mode: 'copy'    
    tag "${test_index}_${rep}"
    conda "${params.repo}/requirements.yml"

    input:
    tuple val(workdir), val(test_index), val(rep)

    output:
    path "results_site_${test_index}_rep_${rep}.txt"

    script:
    """
    python ${params.repo}/train_baselineML.py \
        ${workdir} \
        ${test_index} \
        ${rep} \
        | grep "FINAL OUT" \
        > "results_site_${test_index}_rep_${rep}.txt"
    """
}



process Eval {
    publishDir "${params.workdir}/Out/Baseline_ML/", mode: 'copy'
    tag "eval_baselineML"
    conda "${params.repo}/requirements.yml"

    input:
    tuple val(output_dir), val(plot_title), val(result_files)

    output:
    path "evaluation.pdf"
    
    script:
    """
    python ${params.repo}/evaluate.py \
        ${params.workdir} \
        ${output_dir} \
        "${plot_title}"
    """
}


