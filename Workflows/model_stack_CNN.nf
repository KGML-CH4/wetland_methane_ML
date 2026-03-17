// model-stacking plus CNN



params.each { k, v -> 
    println "- $k = $v"
}
println "=============================\n"

workflow {
    // download MODIS images at fluxnet sites
    Download_MODIS_fluxnet()

    // preprocess TEM and reanalysis data
    Preprocess_TEM()

    // download MODIS images for 0.5 degree grid cells
    ch_a = Channel.of( 1..40 )
    Download_MODIS_global(ch_a)

    // preprocess global MODIS images
    ch_a = Channel.of( 1..XX )
    prep_modis = Prep_MODIS_global_1(ch_a)
    prep_modis = prep_modis.collect()
    prep_modis = Prep_MODIS_global_2(prep_modis)
    prep_modis = Prep_MODIS_global_3(prep_modis)
    prep_modis = prep_modis.collect()
    Prep_MODIS_global_4(prep_modis)

    // preprocess FLUXNET
    Preprocess_FLUXNET()

    // preprocess specific to model-stacking plus CNN
    Preprocess_model()

    // train with baseline ML model                                                               
    ch_a = Channel.of( 1..(params.num_sites) )
    ch_b = Channel.of( 1..(params.num_reps) )
    combined_channel = ch_a.combine(ch_b)
    trained = Train(combined_channel)

    // evaluate                                                                                   
    baselineML
        .collect()
        .map { result_files ->
            tuple("${params.workdir}/Out/Baseline_ML/", "Baseline machine learning", result_files)
        }
        .set { inpt }
    Eval(inpt)

    // preprocess for upscaling
    Preprocess_upscale()

    // predict every grid cell
    Upscale()

    // final upscaling analysis and plots
    Global_plot()
    
}



process Download_MODIS_fluxnet {
    tag "download_modis_fluxnet"
    conda "${params.repo}/requirements.yml"

    output:
    path "modis_images_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/gee_pulldown_FLUXNET.py
    echo "Done." > modis_images_done.txt
    """
}



process Preprocess_TEM() {
    publishDir "${params.workdir}/Out/", mode: 'copy'
    tag "prep_TEM"
    conda "${params.repo}/requirements.yml"

    output:
    path "prep_TEM.sav"

    script:
    """
    python ${params.repo}/Code/preprocess_TEM.py
    """
}



process Download_MODIS_global {
    tag "download_modis_fluxnet_${rep}"
    conda "${params.repo}/requirements.yml"

    intput:
    tuple path "prep_TEM.sav", int(rep)

    output: path "modis_images_done_${rep}.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/gee_pulldown_global.py ${rep}
    echo "Done." > modis_images_done_${rep}.txt
    """
}



process Prep_MODIS_global_1 {
    tag "prep_modis_global_1_${rep}"
    conda "${params.repo}/requirements.yml"

    output: path "modis_prep1_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step1.py ${rep}
    echo "Done." > modis_prep1_done_${rep}.txt
    """
}



process Prep_MODIS_global_2 {
    tag "prep_modis_global_2"
    conda "${params.repo}/requirements.yml"

    output: path "modis_prep2_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step2.py
    echo "Done." > modis_prep2_done.txt
    """
}




process Prep_MODIS_global_3 {
    tag "prep_modis_global_3_${rep}"
    conda "${params.repo}/requirements.yml"

    output: path "modis_prep3_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step3.py ${rep}
    echo "Done." > modis_prep3_done_${rep}.txt
    """
}



process Prep_MODIS_global_4 {
    publishDir "${params.workdir}/Out/MODIS_tiles_TEM/Preprocessed_tiles/", mode: 'copy'
    tag "prep_modis_global_4"
    conda "${params.repo}/requirements.yml"

    output: path "global_SDs.npy"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step4.py ${rep}
    echo "Done." > modis_prep4_done_${rep}.txt
    """
}



process Preprocess_FLUXNET() {
    publishDir "${params.workdir}/Out/", mode: 'copy'
    tag "prep_fluxnet"
    conda "${params.repo}/requirements.yml"

    output:
    path "prep_obs.sav"

    script:
    """
    python ${params.repo}/Code/preprocess_fluxnet.py
    """
}



process Preprocess_model() {
    publishDir "${params.workdir}/Out/", mode: 'copy'
    tag "prep_model"
    conda "${params.repo}/requirements.yml"

    input:
    tuple path "modis_images_done.txt", path "prep_TEM.sav", path "prep_obs.sav"

    output:
    path "prep_model.sav"

    script:
    """
    python ${params.repo}/Code/Model_stacking_CNN/preprocess.py
    """
}



process Train {
    publishDir "${params.workdir}/Out/Model_stack_CNN/", mode: 'copy'    
    tag "train_${test_index}_${rep}"
    conda "${params.repo}/requirements.yml"

    input:
    tuple int(test_index), int(rep), path prep_model.sav

    output:
    path "result_${test_index}_rep_${rep}.txt"

    script:
    """
    python ${params.repo}/Code/Model_stacking_CNN/train.py
        ${test_index} \
        ${rep}
    """
}



process Eval {
    publishDir "${params.workdir}/Out/Model_stack_CNN/", mode: 'copy'
    tag "eval_baselineML"
    conda "${params.repo}/requirements.yml"

    input:
    path "result_${test_index}_rep_${rep}.txt"

    output:
    path "evaluation.pdf"
    
    script:
    """                                                                                                                            
    python ${params.repo}/Code/Model_stacking_CNNevaluate.py \
        ${params.workdir} \                                                                                                        
        ${output_dir} \                                                                                                            
        "${plot_title}"                                                                                                            
    """
}
