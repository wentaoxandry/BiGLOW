#!/bin/bash -u

sourcedatasetdir=/media/wentao/Wentaodisk/projekt/Dataset/Fakeddit
Datasetdir=./Dataset
savedir=./trained
imagedir=./Images
cachedir=./CACHE
iftrainuni=false
unimodaldir=./unimodal_models 
stage=-1
stop_stage=100


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ "$(ls -A $sourcedatasetdir)" ]; then
        echo "Data already downloaded"
    else
        echo "Please download Fakeddit dataset from https://github.com/entitize/Fakeddit"
    fi
    if [[ -f "${Datasetdir}/train.json" ]]; then
	echo "Dataset already processed."
    else
    	python3 local/load_datainfo.py 		\
    		--datasourcedir $sourcedatasetdir 	\
    		--savedir $Datasetdir 	|| exit 1  		
	
    fi    
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    #############   Title Models    ##################
    modal=Title
    if [ "$iftrainuni" = "true" ]; then
    	if [ "$(ls -A ${savedir}/Unimodal/Title/model)" ]; then
    		echo "Title modality already trained"
    	else
    		python3 local/Unimodal/train.py	\
   	  	--datasdir $Datasetdir 		\
   	  	--modal $modal			\
   	  	--lr 5e-6				\
   	  	--BS 32 				\
   	  	--savedir $savedir 	  		\
   	  	--cachedir $cachedir   || exit 1
    	fi
    else
    	if [ "$(ls -A ${Datasetdir}/data/$modal)" ]; then
    		echo "Title modality data already generated"
    	else
    		python3 local/Unimodal/eval_trained_model.py	\
   	  	--datasdir $Datasetdir 		\
   	  	--modal $modal			\
   	  	--BS 32 				\
   	  	--modelsavedir $unimodaldir 	  		\
   	  	--cachedir $cachedir   || exit 1
    	fi
   fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    #############   Image Models    ##################
    modal=Image
    if [ "$iftrainuni" = "true" ]; then
    	if [ "$(ls -A ${savedir}/Unimodal/Image/model)" ]; then
    		echo "Image modality already trained"
    	else
    		python3 local/Unimodal/train.py	\
   	  	--datasdir $Datasetdir 		\
   	  	--modal $modal			\
   	  	--lr 5e-5				\
   	  	--BS 64 				\
   	  	--savedir $savedir 	  		\
   	  	--cachedir $cachedir   || exit 1
      fi
    else
    	if [ "$(ls -A ${Datasetdir}/data/$modal)" ]; then
    		echo "Image modality data already generated"
    	else
    		python3 local/Unimodal/eval_trained_model.py	\
   	  	--datasdir $Datasetdir 		\
   	  	--modal $modal			\
   	  	--BS 64 				\
   	  	--modelsavedir $unimodaldir 	  		\
   	  	--cachedir $cachedir   || exit 1
    	fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ############# global fixed model #################
    if [ "$(ls -A ${savedir}/Multimodal/Global_fixed/GSW_linear)" ]; then
      echo "Global fixed model trained"
    else
    	for type in GSW_linear GSW_logarithmic GSW_logit
    	do
            python3 local/Multimodal/GSW/GSW.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type				\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir  || exit 1	
    	done			
    fi

    if [ "$(ls -A ${savedir}/Subset/Global_fixed/GSW_linear)" ]; then
      echo "Global fixed cv model trained"
    else
    	for type in GSW_linear GSW_logarithmic GSW_logit
    	do
            python3 local/Multimodal/GSW/GSW_subset.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type				\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir  || exit 1	
    	done			
     fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ############# Delta representation model #################
    if [ "$(ls -A ${savedir}/Multimodal/Delta_representation)" ]; then
      echo "Delta representation model trained"
    else
    	for type in Delta_representation Pairwise
    	do
      	python3 local/Multimodal/pairwisefusion/pairwisefusion.py 	\
    		--datasdir $Datasetdir 		\
    		--evaltype $type					\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir    \
        --ifgaussiantest true  || exit 1	
    	done			
    fi

    if [ "$(ls -A ${savedir}/Subset/Delta_representation)" ]; then
      echo "Delta representation cv model trained"
    else
    	for type in Delta_representation Pairwise
    	do
      	python3 local/Multimodal/pairwisefusion/pairwisefusion_subset.py 	\
    		--datasdir $Datasetdir 		\
    		--evaltype $type					\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir  || exit 1	
    	done			
     fi
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ############# Delta representation GMM model #################
    if [ "$(ls -A ${savedir}/Multimodal/Delta_representation_GMM)" ]; then
      echo "Delta representation combined with Gaussian mixture model trained"
    else
    	for type in Delta_representation Pairwise
    	do
      	python3 local/Multimodal/pairwisefusion/pairwisefusion_GMM.py 	\
    		--datasdir $Datasetdir 		\
    		--evaltype $type					\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir  || exit 1	
    	done			
    fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ############# Delta representation GMM model #################
    if [ "$(ls -A ${savedir}/Multimodal/MGL)" ]; then
      echo "Maximum model trained"
    else
        modal=MGL
      	python3 local/Multimodal/pairwisefusion/maximum_gaussian_likelihood.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $modal					\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir    \
        --ifgaussiantest true  || exit 1	
			
    fi
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    ############# Delta representation model #################
    if [ "$(ls -A ${savedir}/Multimodal/representation)" ]; then
      echo "DNN model trained"
    else
    	for type in DSW representation
    	do
      	python3 local/Multimodal/DNN/train.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type					\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir  || exit 1	
    	done			
     fi
     if [ "$(ls -A ${savedir}/Subset/representation)" ]; then
      echo "DNN model cv trained"
    else
    	for type in DSW representation
    	do
      	python3 local/Multimodal/DNN/train_subset.py 	\
    		--datasdir $Datasetdir 		\
    		--modal $type					\
    		--textmodal Title			\
    		--imagemodal Image		\
    		--savedir $savedir  || exit 1	
    	done			
     fi

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ############# Analyzing #################
    if [ "$(ls -A ${imagedir})" ]; then
      echo "Analyzed"
    else
   	  python3 local/Analyse.py 	\
    		--imagedir $imagedir 		\
    		--savedir $savedir  || exit 1			
    fi
fi
