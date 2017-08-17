import csv as csv
import numpy as np
import time 
from pylab import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.svm import LinearSVC 
#from sklearn.linear_model import SGDClassifier
csv_file = csv.reader(open('abc.csv'))
header= csv_file.next()

data = []

for row in csv_file:
	data.append(row)

data = np.array(data)
y=[]


for i in range(0,120000):
	data[i,1] = data[i,1].replace('tcp',"1")
	data[i,1] = data[i,1].replace('udp',"2")
	data[i,1] = data[i,1].replace('icmp',"3")
	data[i,2] = data[i,2].replace('http',"1")
	data[i,2] = data[i,2].replace('smtp',"2")
	data[i,2] = data[i,2].replace('finger',"3")
	data[i,2] = data[i,2].replace('domain_u',"4")
	data[i,2] = data[i,2].replace('auth',"5")
	data[i,2] = data[i,2].replace('telnet',"6")
	data[i,2] = data[i,2].replace('ftp',"7")
	data[i,2] = data[i,2].replace('eco_i',"8")
	data[i,2] = data[i,2].replace('ntp_u',"9")
	data[i,2] = data[i,2].replace('ecr_i',"10")
	data[i,2] = data[i,2].replace('other',"11")
	data[i,2] = data[i,2].replace('private',"12")
	data[i,2] = data[i,2].replace('pop_3',"13")
	data[i,2] = data[i,2].replace('7_data',"14")
	data[i,2] = data[i,2].replace('rje',"15")
	data[i,2] = data[i,2].replace('time',"16")
	data[i,2] = data[i,2].replace('mtp',"17")
	data[i,2] = data[i,2].replace('link',"18")
	data[i,2] = data[i,2].replace('remote_job',"19")
	data[i,2] = data[i,2].replace('gopher',"20")
	data[i,2] = data[i,2].replace('ssh',"21")
	data[i,2] = data[i,2].replace('name',"22")
	data[i,2] = data[i,2].replace('whois',"23")
	data[i,2] = data[i,2].replace('domain',"24")
	data[i,2] = data[i,2].replace('login',"25")
	data[i,2] = data[i,2].replace('imap4',"26")
	data[i,2] = data[i,2].replace('day16',"27")
	data[i,2] = data[i,2].replace('ctf',"28")
	data[i,2] = data[i,2].replace('nntp',"29")
	data[i,2] = data[i,2].replace('shell',"30")
	data[i,2] = data[i,2].replace('IRC',"31")
	data[i,2] = data[i,2].replace('nnsp',"32")
	data[i,2] = data[i,2].replace('1_443',"33")
	data[i,2] = data[i,2].replace('exec',"34")
	data[i,2] = data[i,2].replace('printer',"35")
	data[i,2] = data[i,2].replace('efs',"36")
	data[i,2] = data[i,2].replace('courier',"37")
	data[i,2] = data[i,2].replace('uucp',"38")
	data[i,2] = data[i,2].replace('k25',"39")
	data[i,2] = data[i,2].replace('k30',"40")
	data[i,2] = data[i,2].replace('echo',"41")
	data[i,2] = data[i,2].replace('discard',"42")
	data[i,2] = data[i,2].replace('systat',"43")
	data[i,2] = data[i,2].replace('supdup',"44")
	data[i,2] = data[i,2].replace('iso_tsap',"45")
	data[i,2] = data[i,2].replace('host22s',"46")
	data[i,2] = data[i,2].replace('csnet_ns',"47")
	data[i,2] = data[i,2].replace('pop_2',"48")
	data[i,2] = data[i,2].replace('sunrpc',"49")
	data[i,2] = data[i,2].replace('38_path',"50")
	data[i,2] = data[i,2].replace('netbios_ns',"51")
	data[i,2] = data[i,2].replace('netbios_ssn',"52")
	data[i,2] = data[i,2].replace('netbios_dgm',"53")
	data[i,2] = data[i,2].replace('sql_net',"54")
	data[i,2] = data[i,2].replace('vmnet',"55")
	data[i,2] = data[i,2].replace('bgp',"56")
	data[i,2] = data[i,2].replace('Z39_50',"57")
	data[i,2] = data[i,2].replace('ldap',"58")
	data[i,2] = data[i,2].replace('netstat',"59")
	data[i,2] = data[i,2].replace('urh_i',"60")
	data[i,2] = data[i,2].replace('X11',"61")
	data[i,2] = data[i,2].replace('urp_i',"62")
	data[i,2] = data[i,2].replace('pm_dump',"63")
	data[i,2] = data[i,2].replace('t7_u',"64")
	data[i,2] = data[i,2].replace('tim_i',"65")
	data[i,2] = data[i,2].replace('red_i',"66")
	data[i,3] = data[i,3].replace('SF',"1")
	data[i,3] = data[i,3].replace('S1',"2")
	data[i,3] = data[i,3].replace('REJ',"3")
	data[i,3] = data[i,3].replace('S2',"4")
	data[i,3] = data[i,3].replace('S0',"5")
	data[i,3] = data[i,3].replace('S3',"6")
	data[i,3] = data[i,3].replace('RSTO',"7")
	data[i,3] = data[i,3].replace('RSTR',"8")
	data[i,3] = data[i,3].replace('75',"9")
	data[i,3] = data[i,3].replace('OTH',"10")
	data[i,3] = data[i,3].replace('SH',"11")
	data[i,6] = data[i,6].replace('0',"0")
	data[i,6] = data[i,6].replace('1',"1")
	data[i,11] = data[i,11].replace('0',"0")
	data[i,11] = data[i,11].replace('1',"1")
	data[i,20] = data[i,20].replace('0',"0")
	data[i,20] = data[i,20].replace('1',"1")
	data[i,21] = data[i,21].replace('0',"0")
	data[i,21] = data[i,21].replace('1',"1")
	data[i,41] = data[i,41].replace('normal.',"1")
	data[i,41] = data[i,41].replace('buffer_overflow.',"2")
	data[i,41] = data[i,41].replace('loadmodule.',"3")
	data[i,41] = data[i,41].replace('perl.',"4")
	data[i,41] = data[i,41].replace('neptune.',"5")
	data[i,41] = data[i,41].replace('smurf.',"6")
	data[i,41] = data[i,41].replace('guess_passwd.',"7")
	data[i,41] = data[i,41].replace('pod.',"8")
	data[i,41] = data[i,41].replace('teardrop.',"9")
	data[i,41] = data[i,41].replace('portsweep.',"10")
	data[i,41] = data[i,41].replace('ipsweep.',"11")
	data[i,41] = data[i,41].replace('land.',"12")
	data[i,41] = data[i,41].replace('ftp_write.',"13")
	data[i,41] = data[i,41].replace('back.',"14")
	data[i,41] = data[i,41].replace('imap.',"15")
	data[i,41] = data[i,41].replace('satan.',"16")
	data[i,41] = data[i,41].replace('phf.',"17")
	data[i,41] = data[i,41].replace('nmap.',"18")
	data[i,41] = data[i,41].replace('multihop.',"19")
	data[i,41] = data[i,41].replace('warezmaster.',"20")
	data[i,41] = data[i,41].replace('warezclient.',"21")
	data[i,41] = data[i,41].replace('spy.',"22")
	data[i,41] = data[i,41].replace('rootkit.',"23")
	

#for testing graph
xtrain = data[0:68577,0:41].astype(np.float)
ytrain = data[0:68577,41].astype(np.int)
xtest = data[68578:97968,0:41].astype(np.float)
ytest = data[68578:97968,41].astype(np.int)


#random forest
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(xtrain,ytrain)
p = map(clf.predict,xtest)



#print p

score1='normal.'
score2='buffer_overflow.'
score3='loadmodule.'
score4='perl.'
score5='neptune.'
score6='smurf.'
score7='guess_passwd.'
score8='pod.'
score9='teardrop.'
score10='portsweep.'
score11='ipsweep.'
score12='land.'
score13='ftp_write.'
score14='back.'
score15='imap.'
score16='satan.'
score17='phf.'
score18='nmap.'
score19='multihop.'
score20='warezmaster.'
score21='warezclient.'
score22='spy.'
score23='rootkit.'

index=0
TP=0
TN=0
FP=0
FN=0

for index in range(len(p)):
	if ytest[index] == 1:
		if p[index]==ytest[index]:
			TN=TN+1
	if ytest[index] != 1:
		if p[index]==ytest[index]:
			TP=TP+1
	if ytest[index] == 1:
		if p[index]!=ytest[index]:
			FP=FP+1
	if ytest[index] != 1:
		if p[index]==1:
			FN=FN+1

TP1=0.0
TN1=0.0
FP1=0.0
FN1=0.0
TP2=0.0
TN2=0.0
FP2=0.0
FN2=0.0
TP3=0.0
TN3=0.0
FP3=0.0
FN3=0.0
TP4=0.0
TN4=0.0
FP4=0.0
FN4=0.0
index=0


for index in range(len(p)):
	if ytest[index] == 1:
		if p[index]==ytest[index]:
			TN1=TN1+1
			TN2=TN2+1
			TN3=TN3+1
			TN4=TN4+1
			
	if ytest[index] == 10 or ytest[index] ==11 or ytest[index] ==18 or ytest[index] ==16:
		if p[index]==ytest[index]:
			TP1=TP1+1.0
	if ytest[index] == 1:
		if p[index]==10 or p[index]==11 or p[index]==18 or p[index]==16:
			FP1=FP1+1.0
	if ytest[index] == 10 or ytest[index] == 11 or ytest[index] == 18 or ytest[index] == 16:
		if p[index]==1:
			FN1=FN1+1.0

	if ytest[index] == 5 or ytest[index] ==8 or ytest[index] ==14 or ytest[index] ==12 or ytest[index] ==6 or ytest[index] ==9:
		if p[index]==ytest[index]:
			TP2=TP2+1.0
	if ytest[index] == 1:
		if p[index]==5 or p[index]==8 or p[index]==14 or p[index]==12 or p[index]==6 or p[index]==9:
			FP2=FP2+1.0
	if ytest[index] == 5 or ytest[index] ==8 or ytest[index] ==14 or ytest[index] == 12 or ytest[index] == 6 or ytest[index] == 9:
		if p[index]==1:
			FN2=FN2+1.0

	if ytest[index] == 2 or ytest[index] ==3 or ytest[index] ==4 or ytest[index] ==23:
		if p[index]==ytest[index]:
			TP3=TP3+1.0
	if ytest[index] == 1:
		if p[index]==2 or p[index]==3 or p[index]==4 or p[index]==23:
			FP3=FP3+1.0
	if ytest[index] == 2 or ytest[index] == 3 or ytest[index] == 4 or ytest[index] == 23:
		if p[index]==1:
			FN3=FN3+1.0

	if ytest[index] == 13 or ytest[index] ==15 or ytest[index] ==7 or ytest[index] ==19 or ytest[index] ==17 or ytest[index] ==22 or ytest[index] ==20 or ytest[index] ==21:
		if p[index]==ytest[index]:
			TP4=TP4+1.0
	if ytest[index] == 1:
		if p[index]==13 or p[index]==15 or p[index]==7 or p[index]==19 or p[index]==17 or p[index]==22 or p[index]==20 or p[index]==21:
			FP4=FP4+1.0
	if ytest[index] == 13 or ytest[index] == 15 or ytest[index] == 7 or ytest[index] == 19 or ytest[index] == 17 or ytest[index] == 22 or ytest[index] == 20 or ytest[index] == 21:
		if p[index]==1:
			FN4=FN4+1.0






			
index=0
arr=[]

for index in range(len(p)):
	if p[index] == 1:
		print 'Condition no.',index+1,',prediction is ',score1	
	elif p[index] == 2:
		print 'Condition no.',index+1,',prediction is ',score2
	elif p[index] == 3:
		print 'Condition no.',index+1,',prediction is ',score3
	elif p[index] == 4:
		print 'Condition no.',index+1,',prediction is ',score4
	elif p[index] == 5:
		print 'Condition no.',index+1,',prediction is ',score5
	elif p[index] == 6:
		print 'Condition no.',index+1,',prediction is ',score6
	elif p[index] == 7:
		print 'Condition no.',index+1,',prediction is ',score7
	elif p[index] == 8:
		print 'Condition no.',index+1,',prediction is ',score8
	elif p[index] == 9:
		print 'Condition no.',index+1,',prediction is ',score9
	elif p[index] == 10:
		print 'Condition no.',index+1,',prediction is ',score10
	elif p[index] == 11:
		print 'Condition no.',index+1,',prediction is ',score11
	elif p[index] == 12:
		print 'Condition no.',index+1,',prediction is ',score12
	elif p[index] == 13:
		print 'Condition no.',index+1,',prediction is ',score13
	elif p[index] == 14:
		print 'Condition no.',index+1,',prediction is ',score14
	elif p[index] == 15:
		print 'Condition no.',index+1,',prediction is ',score15
	elif p[index] == 16:
		print 'Condition no.',index+1,',prediction is ',score16
	elif p[index] == 17:
		print 'Condition no.',index+1,',prediction is ',score17
	elif p[index] == 18:
		print 'Condition no.',index+1,',prediction is ',score18
	elif p[index] == 19:
		print 'Condition no.',index+1,',prediction is ',score19
	elif p[index] == 20:
		print 'Condition no.',index+1,',prediction is ',score20
	elif p[index] == 21:
		print 'Condition no.',index+1,',prediction is ',score21
	elif p[index] == 22:
		print 'Condition no.',index+1,',prediction is ',score22
	elif p[index] == 23:
		print 'Condition no.',index+1,',prediction is ',score23
	arr.append(p[index][0]-ytest[index])
	
#print type(p[0])
 
 #MAPE value
arr=np.abs(arr)
print arr
acc=21.21
count=(np.count_nonzero(arr))+0.0
length=len(p)+0.0
acc=((length-count)/length)*100
print 'Accuracy in classification task was ',acc,'%'



#efficiencies
TP=TP+0.0
TN=TN+0.0
FP=FP+0.0
FN=FN+0.0

Accuracy=2.1
Detection_Rate=2.1
False_Alarm=2.1
a=TP+TN
print 'a is ',a
b=TP+TN+FP+FN
print 'b is ',b
c=TP+FP
print 'c is ',c
d=FP+TN
print 'd is ',d
Accuracy = (a/b)
Detection_Rate = TP/c	
False_Alarm = FP/d

print Accuracy*100
print Detection_Rate*100
print False_Alarm*100
print '....................................'

TP1=TP1+0.0
TN1=TN1+0.0
FP1=FP1+0.0
FN1=FN1+0.0

print 'TP is',TP1
print 'TN is',TN1
print 'FP is',FP1
print 'FN is',FN1



a=TP1+TN1
b=TP1+TN1+FP1+FN1
c=TP1+FP1
d=FP1+TN1
Accuracy1a = (a/b)
Detectionrate1a = TP1/c
False_alarm1a = FP1 /d 
print 'accuracy for probe',Accuracy1a*100
print 'detection rate for probe',Detectionrate1a*100
print 'false alarm for probe',False_alarm1a*100
print '.....................................'
TP2=TP2+0.0
TN2=TN2+0.0
FP2=FP2+0.0
FN2=FN2+0.0

print 'TP is',TP2
print 'TN is',TN2
print 'FP is',FP2
print 'FN is',FN2


a=TP2+TN2
b=TP2+TN2+FP2+FN2
c=TP2+FP2
d=FP2+TN2
Accuracy1b = (a/b)
Detectionrate1b = TP2/c
False_alarm1b = FP2 /d 
print 'accuracy for dos',Accuracy1b*100
print 'detection rate for dos',Detectionrate1b*100
print 'false alarm for dos',False_alarm1b*100
print '.......................................'
TP3=TP3+0.0
TN3=TN3+0.0
FP3=FP3+0.0
FN3=FN3+0.0

print 'TP is',TP3
print 'TN is',TN3
print 'FP is',FP3
print 'FN is',FN3

a=TP3+TN3
b=TP3+TN3+FP3+FN3
c=TP3+FP3
d=FP3+TN3
Accuracy1c = (a/b)
Detectionrate1c = ((TP3+0.0)/(c+0.0))
False_alarm1c = FP3/d 
print 'accuracy for u2r',Accuracy1c*100
print 'detection rate for u2r',Detectionrate1c*100
print 'false alarm for u2r',False_alarm1c*100
print '.........................................'
TP4=TP4+0.0
TN4=TN4+0.0
FP4=FP4+0.0
FN4=FN4+0.0

print 'TP is',TP4
print 'TN is',TN4
print 'FP is',FP4
print 'FN is',FN4

a=TP4+TN4
b=TP4+TN4+FP4+FN4
c=TP4+FP4
d=FP4+TN4
Accuracy1d = (a/b)
Detectionrate1d = TP4/c
False_alarm1d = FP4/d 
print 'accuracy for r2l',Accuracy1d*100
print 'detection rate for r2l',Detectionrate1d*100
print 'false alarm for r2l',False_alarm1d*100
show()
time.sleep(50)
#---------------------------------------------------------------------------------------------------------------

#decision trees
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain,ytrain)
p = map(clf.predict,xtest)

#print p

score1='normal.'
score2='buffer_overflow.'
score3='loadmodule.'
score4='perl.'
score5='neptune.'
score6='smurf.'
score7='guess_passwd.'
score8='pod.'
score9='teardrop.'
score10='portsweep.'
score11='ipsweep.'
score12='land.'
score13='ftp_write.'
score14='back.'
score15='imap.'
score16='satan.'
score17='phf.'
score18='nmap.'
score19='multihop.'
score20='warezmaster.'
score21='warezclient.'
score22='spy.'
score23='rootkit.'

index=0
TP=0
TN=0
FP=0
FN=0

for index in range(len(p)):
	if ytest[index] == 1:
		if p[index]==ytest[index]:
			TN=TN+1
	if ytest[index] != 1:
		if p[index]==ytest[index]:
			TP=TP+1
	if ytest[index] == 1:
		if p[index]!=ytest[index]:
			FP=FP+1
	if ytest[index] != 1:
		if p[index]==1:
			FN=FN+1

TP1=0
TN1=0
FP1=0
FN1=0
TP2=0
TN2=0
FP2=0
FN2=0
TP3=0
TN3=0
FP3=0
FN3=0
TP4=0
TN4=0
FP4=0
FN4=0
index=0


for index in range(len(p)):
	if ytest[index] == 1:
		if p[index]==ytest[index]:
			TN1=TN1+1
			TN2=TN2+1
			TN3=TN3+1
			TN4=TN4+1
			
	if ytest[index] == 10 or ytest[index] ==11 or ytest[index] ==18 or ytest[index] ==16:
		if p[index]==ytest[index]:
			TP1=TP1+1
	if ytest[index] == 1:
		if p[index]==10 or p[index]==11 or p[index]==18 or p[index]==16:
			FP1=FP1+1
	if ytest[index] == 10 or ytest[index] == 11 or ytest[index] == 18 or ytest[index] == 16:
		if p[index]==1:
			FN1=FN1+1

	if ytest[index] == 5 or ytest[index] ==8 or ytest[index] ==14 or ytest[index] ==12 or ytest[index] ==6 or ytest[index] ==9:
		if p[index]==ytest[index]:
			TP2=TP2+1
	if ytest[index] == 1:
		if p[index]==5 or p[index]==8 or p[index]==14 or p[index]==12 or p[index]==6 or p[index]==9:
			FP2=FP2+1
	if ytest[index] == 5 or ytest[index] ==8 or ytest[index] ==14 or ytest[index] == 12 or ytest[index] == 6 or ytest[index] == 9:
		if p[index]==1:
			FN2=FN2+1

	if ytest[index] == 2 or ytest[index] ==3 or ytest[index] ==4 or ytest[index] ==23:
		if p[index]==ytest[index]:
			TP3=TP3+1
	if ytest[index] == 1:
		if p[index]==2 or p[index]==3 or p[index]==4 or p[index]==23:
			FP3=FP3+1
	if ytest[index] == 2 or ytest[index] == 3 or ytest[index] == 4 or ytest[index] == 23:
		if p[index]==1:
			FN3=FN3+1

	if ytest[index] == 13 or ytest[index] ==15 or ytest[index] ==7 or ytest[index] ==19 or ytest[index] ==17 or ytest[index] ==22 or ytest[index] ==20 or ytest[index] ==21:
		if p[index]==ytest[index]:
			TP4=TP4+1
	if ytest[index] == 1:
		if p[index]==13 or p[index]==15 or p[index]==7 or p[index]==19 or p[index]==17 or p[index]==22 or p[index]==20 or p[index]==21:
			FP4=FP4+1
	if ytest[index] == 13 or ytest[index] == 15 or ytest[index] == 7 or ytest[index] == 19 or ytest[index] == 17 or ytest[index] == 22 or ytest[index] == 20 or ytest[index] == 21:
		if p[index]==1:
			FN4=FN4+1






			
index=0
arr=[]

for index in range(len(p)):
	if p[index] == 1:
		print 'Condition no.',index+1,',prediction is ',score1	
	elif p[index] == 2:
		print 'Condition no.',index+1,',prediction is ',score2
	elif p[index] == 3:
		print 'Condition no.',index+1,',prediction is ',score3
	elif p[index] == 4:
		print 'Condition no.',index+1,',prediction is ',score4
	elif p[index] == 5:
		print 'Condition no.',index+1,',prediction is ',score5
	elif p[index] == 6:
		print 'Condition no.',index+1,',prediction is ',score6
	elif p[index] == 7:
		print 'Condition no.',index+1,',prediction is ',score7
	elif p[index] == 8:
		print 'Condition no.',index+1,',prediction is ',score8
	elif p[index] == 9:
		print 'Condition no.',index+1,',prediction is ',score9
	elif p[index] == 10:
		print 'Condition no.',index+1,',prediction is ',score10
	elif p[index] == 11:
		print 'Condition no.',index+1,',prediction is ',score11
	elif p[index] == 12:
		print 'Condition no.',index+1,',prediction is ',score12
	elif p[index] == 13:
		print 'Condition no.',index+1,',prediction is ',score13
	elif p[index] == 14:
		print 'Condition no.',index+1,',prediction is ',score14
	elif p[index] == 15:
		print 'Condition no.',index+1,',prediction is ',score15
	elif p[index] == 16:
		print 'Condition no.',index+1,',prediction is ',score16
	elif p[index] == 17:
		print 'Condition no.',index+1,',prediction is ',score17
	elif p[index] == 18:
		print 'Condition no.',index+1,',prediction is ',score18
	elif p[index] == 19:
		print 'Condition no.',index+1,',prediction is ',score19
	elif p[index] == 20:
		print 'Condition no.',index+1,',prediction is ',score20
	elif p[index] == 21:
		print 'Condition no.',index+1,',prediction is ',score21
	elif p[index] == 22:
		print 'Condition no.',index+1,',prediction is ',score22
	elif p[index] == 23:
		print 'Condition no.',index+1,',prediction is ',score23
	arr.append(p[index][0]-ytest[index])
	
#print type(p[0])
 
 #MAPE value
arr=np.abs(arr)
print arr
acc=21.21
count=(np.count_nonzero(arr))+0.0
length=len(p)+0.0
acc=((length-count)/length)*100
print 'Accuracy in classification task was ',acc,'%'



#efficiencies
TP=TP+0.0
TN=TN+0.0
FP=FP+0.0
FN=FN+0.0

Accuracy=2.1
Detection_Rate=2.1
False_Alarm=2.1
a=TP+TN
#print 'a is ',a
b=TP+TN+FP+FN
#print 'b is ',b
c=TP+FP
#print 'c is ',c
d=FP+TN
#print 'd is ',d
Accuracy = (a/b)
Detection_Rate = TP/c	
False_Alarm = FP/d

print Accuracy*100
print Detection_Rate*100
print False_Alarm*100

TP1=TP1+0.0
TN1=TN1+0.0
FP1=FP1+0.0
FN1=FN1+0.0

a=TP1+TN1
b=TP1+TN1+FP1+FN1
Accuracy2a = (a/b)
Accuracy2a = Accuracy2a *100
print 'accuracy for probe',Accuracy2a

TP2=TP2+0.0
TN2=TN2+0.0
FP2=FP2+0.0
FN2=FN2+0.0

a=TP2+TN2
b=TP2+TN2+FP2+FN2
Accuracy2b = (a/b)
Accuracy2b = Accuracy2b *100
print 'accuracy for dos',Accuracy2b


TP3=TP3+0.0
TN3=TN3+0.0
FP3=FP3+0.0
FN3=FN3+0.0

a=TP3+TN3
b=TP3+TN3+FP3+FN3
Accuracy2c = (a/b)
Accuracy2c = Accuracy2c *100
print 'accuracy for u2r',Accuracy2c

TP4=TP4+0.0
TN4=TN4+0.0
FP4=FP4+0.0
FN4=FN4+0.0

a=TP4+TN4
b=TP4+TN4+FP4+FN4
Accuracy2d = (a/b)
Accuracy2d = Accuracy2d *100
print 'accuracy for r2l',Accuracy2d
show()
time.sleep(50)
#-----------------------------------------------------------------------------------------

#nearest neighbours
clf = NearestCentroid()
clf.fit(xtrain,ytrain)
p = map(clf.predict,xtest)

#print p

score1='normal.'
score2='buffer_overflow.'
score3='loadmodule.'
score4='perl.'
score5='neptune.'
score6='smurf.'
score7='guess_passwd.'
score8='pod.'
score9='teardrop.'
score10='portsweep.'
score11='ipsweep.'
score12='land.'
score13='ftp_write.'
score14='back.'
score15='imap.'
score16='satan.'
score17='phf.'
score18='nmap.'
score19='multihop.'
score20='warezmaster.'
score21='warezclient.'
score22='spy.'
score23='rootkit.'

index=0
TP=0
TN=0
FP=0
FN=0

for index in range(len(p)):
	if ytest[index] == 1:
		if p[index]==ytest[index]:
			TN=TN+1
	if ytest[index] != 1:
		if p[index]==ytest[index]:
			TP=TP+1
	if ytest[index] == 1:
		if p[index]!=ytest[index]:
			FP=FP+1
	if ytest[index] != 1:
		if p[index]==1:
			FN=FN+1

TP1=0
TN1=0
FP1=0
FN1=0
TP2=0
TN2=0
FP2=0
FN2=0
TP3=0
TN3=0
FP3=0
FN3=0
TP4=0
TN4=0
FP4=0
FN4=0
index=0


for index in range(len(p)):
	if ytest[index] == 1:
		if p[index]==ytest[index]:
			TN1=TN1+1
			TN2=TN2+1
			TN3=TN3+1
			TN4=TN4+1
			
	if ytest[index] == 10 or ytest[index] ==11 or ytest[index] ==18 or ytest[index] ==16:
		if p[index]==ytest[index]:
			TP1=TP1+1
	if ytest[index] == 1:
		if p[index]==10 or p[index]==11 or p[index]==18 or p[index]==16:
			FP1=FP1+1
	if ytest[index] == 10 or ytest[index] == 11 or ytest[index] == 18 or ytest[index] == 16:
		if p[index]==1:
			FN1=FN1+1

	if ytest[index] == 5 or ytest[index] ==8 or ytest[index] ==14 or ytest[index] ==12 or ytest[index] ==6 or ytest[index] ==9:
		if p[index]==ytest[index]:
			TP2=TP2+1
	if ytest[index] == 1:
		if p[index]==5 or p[index]==8 or p[index]==14 or p[index]==12 or p[index]==6 or p[index]==9:
			FP2=FP2+1
	if ytest[index] == 5 or ytest[index] ==8 or ytest[index] ==14 or ytest[index] == 12 or ytest[index] == 6 or ytest[index] == 9:
		if p[index]==1:
			FN2=FN2+1

	if ytest[index] == 2 or ytest[index] ==3 or ytest[index] ==4 or ytest[index] ==23:
		if p[index]==ytest[index]:
			TP3=TP3+1
	if ytest[index] == 1:
		if p[index]==2 or p[index]==3 or p[index]==4 or p[index]==23:
			FP3=FP3+1
	if ytest[index] == 2 or ytest[index] == 3 or ytest[index] == 4 or ytest[index] == 23:
		if p[index]==1:
			FN3=FN3+1

	if ytest[index] == 13 or ytest[index] ==15 or ytest[index] ==7 or ytest[index] ==19 or ytest[index] ==17 or ytest[index] ==22 or ytest[index] ==20 or ytest[index] ==21:
		if p[index]==ytest[index]:
			TP4=TP4+1
	if ytest[index] == 1:
		if p[index]==13 or p[index]==15 or p[index]==7 or p[index]==19 or p[index]==17 or p[index]==22 or p[index]==20 or p[index]==21:
			FP4=FP4+1
	if ytest[index] == 13 or ytest[index] == 15 or ytest[index] == 7 or ytest[index] == 19 or ytest[index] == 17 or ytest[index] == 22 or ytest[index] == 20 or ytest[index] == 21:
		if p[index]==1:
			FN4=FN4+1






			
index=0
arr=[]

for index in range(len(p)):
	if p[index] == 1:
		print 'Condition no.',index+1,',prediction is ',score1	
	elif p[index] == 2:
		print 'Condition no.',index+1,',prediction is ',score2
	elif p[index] == 3:
		print 'Condition no.',index+1,',prediction is ',score3
	elif p[index] == 4:
		print 'Condition no.',index+1,',prediction is ',score4
	elif p[index] == 5:
		print 'Condition no.',index+1,',prediction is ',score5
	elif p[index] == 6:
		print 'Condition no.',index+1,',prediction is ',score6
	elif p[index] == 7:
		print 'Condition no.',index+1,',prediction is ',score7
	elif p[index] == 8:
		print 'Condition no.',index+1,',prediction is ',score8
	elif p[index] == 9:
		print 'Condition no.',index+1,',prediction is ',score9
	elif p[index] == 10:
		print 'Condition no.',index+1,',prediction is ',score10
	elif p[index] == 11:
		print 'Condition no.',index+1,',prediction is ',score11
	elif p[index] == 12:
		print 'Condition no.',index+1,',prediction is ',score12
	elif p[index] == 13:
		print 'Condition no.',index+1,',prediction is ',score13
	elif p[index] == 14:
		print 'Condition no.',index+1,',prediction is ',score14
	elif p[index] == 15:
		print 'Condition no.',index+1,',prediction is ',score15
	elif p[index] == 16:
		print 'Condition no.',index+1,',prediction is ',score16
	elif p[index] == 17:
		print 'Condition no.',index+1,',prediction is ',score17
	elif p[index] == 18:
		print 'Condition no.',index+1,',prediction is ',score18
	elif p[index] == 19:
		print 'Condition no.',index+1,',prediction is ',score19
	elif p[index] == 20:
		print 'Condition no.',index+1,',prediction is ',score20
	elif p[index] == 21:
		print 'Condition no.',index+1,',prediction is ',score21
	elif p[index] == 22:
		print 'Condition no.',index+1,',prediction is ',score22
	elif p[index] == 23:
		print 'Condition no.',index+1,',prediction is ',score23
	arr.append(p[index][0]-ytest[index])
	
#print type(p[0])
 
 #MAPE value
arr=np.abs(arr)
print arr
acc=21.21
count=(np.count_nonzero(arr))+0.0
length=len(p)+0.0
acc=((length-count)/length)*100
print 'Accuracy in classification task was ',acc,'%'


#show()

#efficiencies
TP=TP+0.0
TN=TN+0.0
FP=FP+0.0
FN=FN+0.0

Accuracy=2.1
Detection_Rate=2.1
False_Alarm=2.1
a=TP+TN
print 'a is ',a
b=TP+TN+FP+FN
print 'b is ',b
c=TP+FP
print 'c is ',c
d=FP+TN
print 'd is ',d
Accuracy = (a/b)
Detection_Rate = TP/c	
False_Alarm = FP/d

print Accuracy*100
print Detection_Rate*100
print False_Alarm*100

TP1=TP1+0.0
TN1=TN1+0.0
FP1=FP1+0.0
FN1=FN1+0.0

a=TP1+TN1
b=TP1+TN1+FP1+FN1
Accuracy3a = (a/b)
Accuracy3a = Accuracy3a *100
print 'accuracy for probe',Accuracy3a

TP2=TP2+0.0
TN2=TN2+0.0
FP2=FP2+0.0
FN2=FN2+0.0

a=TP2+TN2
b=TP2+TN2+FP2+FN2
Accuracy3b = (a/b)
Accuracy3b = Accuracy3b *100
print 'accuracy for dos',Accuracy3b


TP3=TP3+0.0
TN3=TN3+0.0
FP3=FP3+0.0
FN3=FN3+0.0

a=TP3+TN3
b=TP3+TN3+FP3+FN3
Accuracy3c = (a/b)
Accuracy3c = Accuracy3c *100
print 'accuracy for u2r',Accuracy3c

TP4=TP4+0.0
TN4=TN4+0.0
FP4=FP4+0.0
FN4=FN4+0.0

a=TP4+TN4
b=TP4+TN4+FP4+FN4
Accuracy3d = (a/b)
Accuracy3d = Accuracy3d *100
print 'accuracy for r2l',Accuracy3d
show()

