# pi0.def で %runscript だけ更新する
apptainer build --force --section runscript pi0.sif pi0.def

#sandbox作成
#apptainer build --fakeroot --sandbox pi0.sandbox pi0.def
#sandbox更新
#apptainer build --fakeroot --update pi0.sandbox pi0.def
#sandbox->sif
#apptainer build --fakeroot pi0.sif pi0.sandbox

