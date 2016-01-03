# Note: 
# After running different individual methods, 
# you need firstly write the result path of these methods into file "data/{inputfile=data_source.txt}". 
# The format of {inputfile}: first line is the path of ture label of valid date, 
# following lines are the path of different individual methods, one line one method.

overwrite=1

# combine result from different individul methods
dev_collection=msr2013dev
inputfile=individual_result_pathes.txt
resultname=qid.img.lable.feature.txt

python combine_result.py $dev_collection --inputfile $inputfile --resultname $resultname --overwrite $overwrite


# weight optimization
inputeFile=$resultname
weightFile=optimized_wights.txt
resultFile=fianl.result.txt

#weight optimization
python weightOptimization.py --inputeFile $inputeFile --weightFile $weightFile --overwrite $overwrite

#relevance fusion
python relevanceFusion.py --inputeFile $inputeFile --weightFile $weightFile --resultFile $resultFile --overwrite $overwrite
