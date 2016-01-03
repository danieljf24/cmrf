
dev_collection=msr2013dev
train_collection=msr2013train
overwrite=1

# image2text
method=ia #i2t
feature=ruccaffefc7.imagenet
nnimagefile=id.100nn.dis.txt
imageclickfile=image.clicked.txt

python cmrf_main_all.py $train_collection $dev_collection --method $method --feature $feature --nnimagefile $nnimagefile --imageclickfile $imageclickfile --overwrite $overwrite


# text2image
method=ta #t2i
feature=ruccaffefc7.imagenet
nnqueryfile=qid.100nn.score.txt
queryclickfile=query.clicked.txt

python cmrf_main_all.py $train_collection $dev_collection --method $method --feature $feature --nnqueryfile $nnqueryfile --queryclickfile $queryclickfile --overwrite $overwrite


# conse
method=conse
corpus=flickr
word2vec=vec200flickr25m
feature=ruccaffeprob.imagenet
label_source=ilsvrc12

python cmrf_main_all.py $train_collection $dev_collection --method $method --feature $feature --corpus $corpus --word2vec $word2vec --label_source $label_source  --overwrite $overwrite


# psi
method=psi
feature=ruccaffefc7.imagenet
psi_model_name=trained_psi_model.pkl

python cmrf_main_all.py $train_collection $dev_collection --method $method --feature $feature --psi_model_name $psi_model_name  --overwrite $overwrite


# devise
method=devise
feature=ruccaffefc7.imagenet
devise_model_name=trained_devise_model.pkl

python cmrf_main_all.py $train_collection $dev_collection --method $method --feature $feature --devise_model_name $devise_model_name  --overwrite $overwrite




# Parzen window
# method=pw
# sigma=0.5
# feature=ruccaffefc7.imagenet.L2
# python parzenWindow.py $dev_collection --method $method --feature $feature --sigma $sigma --overwrite $overwrite
