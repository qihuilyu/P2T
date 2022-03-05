#!/bin/bash
set -e 

sshusername='qlyu'
sshpassword='mileylqh'
#COMPUTE_HOSTS=(shenggpu1 shenggpu2 '10.44.114.117' ketosuccess '10.44.114.215' microway thinkmate1tb ryan-thinkstation-d30 daili-ThinkStation-D30 shenglab2 shenglab5 shenglab7 shenglab10-desktop shenglab6 Qihui)
COMPUTE_HOSTS=(shenggpu1 shenggpu2 '10.44.114.117' ketosuccess '10.44.114.215' microway thinkmate1tb ryan-thinkstation-d30 shenglab5 shenglab7 shenglab10-desktop shenglab2 Qihui)
#COMPUTE_HOSTS=(daili-ThinkStation-D30 shenglab4)
projfolder='/media/raid1/qlyu/PairProd/'
projName='pairprod'

cd $projfolder'code'
echo $sshpassword | sudo -S ./docker-build.sh --regen -f
cd 'dosecalc'
echo $sshpassword | sudo -S ./simpledose bundle-computeserver
echo $sshpassword | sudo -S ./simpledose restart

for host in ${COMPUTE_HOSTS[@]}; do
	sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $sshusername@${host}  << EOF
	cd ~
	echo $host
	rm computeserver.bundle.tar -f
	rm simpledose_compute -rf	
	echo 'computeserver.bundle.tar removed or not exists...'
	echo $sshpassword | sudo -S docker kill 'dosecalc-'$projName'-computeserver'
	echo 'killed (or not exist) dosecalc-'$projName'-computeserver' on $host
	exit
EOF
	sshpass -p $sshpassword scp computeserver.bundle.tar $sshusername@${host}:~ 
	echo 'new computeserver.bundle.tar sent to '${host}'...'
	sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $sshusername@${host}  << EOF
	echo $sshpassword | sudo -S mkdir $projfolder'dbdata' -p
	cd ~
	tar -xvf computeserver.bundle.tar
	cd simpledose_compute
	echo $sshpassword | sudo -S ./simpledose start-computeserver
	echo 'new computeserver is set at '${host}'...'
	exit
EOF
done




