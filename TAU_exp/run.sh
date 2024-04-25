#!/bin/bash -u


Datasetdir=./Dataset
sourcedatasetdir=/media/wentao/Wentaodisk/projekt/Dataset/TAU_Urban_Audio-Visual_Scenes_2021/
Featuredir=${Datasetdir}/Features
savedir=./trained
imagedir=./Images
cachedir=./CACHE
iftrainuni=false
unimodaldir=./unimodal_models 
stage=0
stop_stage=100

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    mkdir -p ${Datasetdir}/Development
    mkdir -p ${Datasetdir}/Evaluation
    if [ "$(ls -A $Datasetdir/Development)" ]; then
        echo "Data already downloaded"
    else
        echo "Download and save in $Datasetdir/Development folder. Downloadlink is https://zenodo.org/record/4477542"
        echo "Download and save in $Datasetdir/Evaluation folder. Downloadlink is https://zenodo.org/record/4767103"
    fi
    if [ -d "$sourcedatasetdir" ]; then
    	echo "$sourcedatasetdir exists."
    else
    	mkdir -p $sourcedatasetdir/Development
    	unzip $Datasetdir/Development/\*.zip -d $sourcedatasetdir/Development
    	mkdir -p $sourcedatasetdir/Evaluation
    	unzip $Datasetdir/Evaluation/\*.zip -d $sourcedatasetdir/Evaluation
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # creat json file for training
    # Since the Evaluation set does not have label, we use a part of data from Develipment set as inter Evaluation set. Download the proposed sub-set list from https://github.com/shanwangshan/TAU-urban-audio-visual-scenes/tree/main/create_data/evaluation_setup rename train.csv as Development.csv and val.csv as Evaluation.csv in $Datasetdir
    ##################### error file: audio/airport-lyon-1095-40158 only audio no video ########################################
    if [ "$(ls -A ${Datasetdir}/test.json)" ]; then
        echo "Dataset already extracted"
    else
    	python3 local/load_datainfo.py --datadir $Datasetdir \
                                     --sourcedatadir $sourcedatasetdir || exit 1;
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train audio model
    modal=audio
    if [ "$iftrainuni" = "true" ]; then
    	if [ "$(ls -A ${savedir}/Unimodal/${modal}/results)" ]; then
    		echo "audio-only model already pretrained"
    	else
    		python3 local/unimodal/train.py --datadir ${Datasetdir} \
    					--featdir ${Featuredir}	\
    					--savedir ${savedir}  \
                                     --modal $modal        \
                                     --cashedir $cachedir || exit 1;
    	fi
    else
    	if [ "$(ls -A ${Datasetdir}/data/$modal)" ]; then
    		echo "audio modality data already generated"
    	else
    		python3 local/unimodal/eval_trained_model.py	\
   	  	--datadir $Datasetdir 		\
   	  	--featdir ${Featuredir}	\
   	  	--modal $modal			\
   	  	--modelsavedir $unimodaldir 	\
   	  	--cachedir $cachedir   || exit 1
    	fi
    	
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # train video model
    
    # As we mentioned in our paper, we extract 10 frames from video for classification
    if [ "$(ls -A ${Datasetdir}/Features/test)" ]; then
    	echo "extract video frames"
    else

    python3 local/unimodal/extract_video.py --datadir ${Datasetdir} \
    					--featdir ${Featuredir}	|| exit 1;
    fi
    modal=video
    if [ "$iftrainuni" = "true" ]; then
    	if [ "$(ls -A ${savedir}/Unimodal/${modal}/results)" ]; then
    		echo "video-only model already pretrained"
    	else

    		python3 local/unimodal/train.py --datadir ${Datasetdir} \
    					--featdir ${Featuredir}	\
    					--savedir ${savedir}  \
                                     --modal $modal        \
                                     --cashedir $cachedir || exit 1;
    	fi
    else
    	if [ "$(ls -A ${Datasetdir}/data/$modal)" ]; then
    		echo "video modality data already generated"
    	else
    		python3 local/unimodal/eval_trained_model.py	\
   	  	--datadir $Datasetdir 		\
   	  	--featdir ${Featuredir}	\
   	  	--modal $modal			\
   	  	--modelsavedir $unimodaldir 	\
   	  	--cachedir $cachedir   || exit 1
    	fi
    	
    fi
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ############# Delta representation model #################
    if [ "$(ls -A ${savedir}/Multimodal/Delta_representation)" ]; then
      echo "Delta representation model trained"
    else
    	for type in Delta_representation Pairwise
    	do
      	python3 local/multimodal/pairwisefusion/pairwisefusion.py 	\
    		--datasdir $Datasetdir 		\
    		--evaltype $type					\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir $savedir  || exit 1	
    	done			
    fi

    if [ "$(ls -A ${savedir}/Subset/Delta_representation)" ]; then
      echo "Delta representation cv model trained"
    else
    	for type in Delta_representation Pairwise
    	do
      	python3 local/multimodal/pairwisefusion/pairwisefusion_subset.py 	\
    		--datasdir $Datasetdir 		\
    		--evaltype $type					\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir $savedir  || exit 1	
    	done			
     fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ############# Delta representation model #################
    if [ "$(ls -A ${savedir}/Multimodal/MGL)" ]; then
      echo "Maximum model trained"
    else
    	type=MGL
      	python3 local/multimodal/pairwisefusion/pairwisefusion_maximul_likelihood.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type					\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir $savedir    \
        	--ifgaussiantest true  || exit 1	
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ############# Delta representation combined with Gaussian mixture model #################
    if [ "$(ls -A ${savedir}/Multimodal/Delta_representation_GMM)" ]; then
      	echo "Delta representation combined with Gaussian mixture model trained"
    else
      	for type in Delta_representation Pairwise
    	do
            python3 local/multimodal/pairwisefusion/pairwisefusion_GMM.py 	\
    		--datasdir $Datasetdir 		\
    		--evaltype $type				\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir ${savedir}  || exit 1	
	done	
    fi
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ############# global fixed model #################
    if [ "$(ls -A ${savedir}/Multimodal/Global_fixed/Global_linear)" ]; then
      echo "Global fixed model trained"
    else
    	for type in Global_linear Global_logarithmic Global_logit
    	do
            python3 local/multimodal/GSW/GSW.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type				\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir ${savedir}  || exit 1	
    	done			
    fi

    if [ "$(ls -A ${savedir}/Subset/Global_fixed/Global_linear)" ]; then
      echo "Global fixed cv model trained"
    else
    	for type in Global_linear Global_logarithmic Global_logit
    	do
            python3 local/multimodal/GSW/GSW_subset.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type				\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir $savedir  || exit 1	
    	done			
     fi


fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    ############# Delta representation model #################
    if [ "$(ls -A ${savedir}/Multimodal/representation)" ]; then
      echo "DNN model trained"
    else
    	for type in DSW representation
    	do
      	python3 local/multimodal/DNN/train.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type					\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir $savedir  || exit 1	
    	done			
     fi

    if [ "$(ls -A ${savedir}/Subset/representation)" ]; then
      echo "DNN model trained"
    else
    	for type in DSW representation
    	do
      	python3 local/multimodal/DNN/train_subset.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type					\
    		--videomodal video			\
    		--audiomodal audio		\
    		--savedir $savedir  || exit 1	
    	done			
    fi
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    ############# Analyzing #################
    if [ "$(ls -A ${imagedir})" ]; then
      echo "Analyzed"
    else
   	  python3 local/Analyse.py 	\
    		--imagedir $imagedir 		\
    		--savedir $savedir  || exit 1			
    fi
fi





