import math
import copy

#########################
########Functions########
#########################
def readData():
	data=[]
	fp=open("data.txt")
	print "--------Rows--------\n", fp.readline()

	for temp in fp:
		data.append(temp.split())
		
	fp.close()
	return data

def makeColumns(data):
	columns=[]
	for i in range(len(data)):
		columns.append([])

	for i in data:
		for key, value in enumerate(i):
			columns[key].append(value)
	columns=filter(None, columns)
	return columns
########Functions########



##############################
########Initialization########
##############################
data=readData() #데이터를 읽어와 data에 적재
k=3 #k값
target=[43.0, 5.0, 57.0, "?"] #타겟데이터
########Initialize########





#############################
########Normalization########
#############################
def normalization(array, target):
	minvalue, maxvalue=float(min(array)), float(max(array)) #정규화를 위한 최초값과 최대값
	return map(lambda x: (x-minvalue)/(maxvalue-minvalue), array), (target-minvalue)/(maxvalue-minvalue) #정규화된 데이터와 타겟데이터를 리턴

columns=makeColumns(data) #읽어온 데이터를 필드별로 분류하여 columns 리스트에 삽입 ex) [war, kick, kiss], [war, kick, kiss],[war, kick, kiss] -> [war, war, war], [kick, kick, kick], [kiss, kiss, kiss]

for key, value in enumerate(columns):
	if value[key][0].isdigit(): # [class]가 아니라면(string이므로 제외)
		columns[key], target[key]=normalization(map(int, value), target[key]) #정규화된 리스트와 정규화된 타겟데이터로 값을 변경
########Normalization#####





####################################
########Distance calculation########
####################################		
def calDistance(columns, target):
	distances=[] #거리를 저장할 list

	for i in range(len(columns[0])): # number of elements in a column
			sum=0
			for j in range(len(columns)): # number of elements in a row 여러개의 컬럼을 한줄씩 내려감
					if type(target[j])!=str: #정규화에서와 마찬가지로 string이므로 제외
						sum=sum+pow((columns[j][i]-target[j]), 2) #거리 계산
			distances.append(math.sqrt(sum)) #계산된 거리를 distances 리스트에 저장

	return distances #거리가 저장된 리스트를 리턴

distances=calDistance(columns, target)
########Distance calculation########





#################################
########Result evaluation########
#################################
nearest=[] #k개의 가장 가까운 거리를 저장할 리스트
temp=copy.deepcopy(distances) #단순히 pop을 쓰기 위함

for i in range(k): #k개의 거리를 뽑아냄
	nearest.append(distances.index(min(temp))) #작은 순서대로 push
	temp.pop(temp.index(min(temp))) #pop
del temp

kNN={} #k개의 거리를 순서대로 정렬하여 class를 찾아냄
for i in nearest: #k개의 nearest 리스트 안에서 히스토그램(?) 카운트를 함(다수결)
	if kNN.has_key(data[i][3]):
		kNN[data[i][3]]=kNN[data[i][3]]+1
	else:
		kNN[data[i][3]]=1

answer=kNN.keys()[kNN.values().index(max(kNN.values()))] #가장 카운트가 많은 nearest의 값이 output
print "Answer: %s" %(answer) 
########Result evaluation########
