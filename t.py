#匯入資料
l = []
while True:
    num = input()
    if num == 'BREAK':
        break
    else:
        items = num.split(" ")
        l.append(items)

name = []
for i in range(len(l)):
    if l[i][0] not in name:
        name.append(l[i][0])    

com = []
for i in range(len(name)):
    pair = []
    pair.append(name[i])
    com.append(pair)

    for j in range(len(l)):
        if l[j][0] == name[i]:
            while True:
                #若名稱對應的value有效
                try:
                    l[j][1] = float(l[j][1])
                    pair.append(l[j][1])
                    break
                #若名稱對應的value無效
                except ValueError:
                    break
                
    #若名稱對應的value皆無效
    if len(pair) == 1:
        com.remove(pair)            
            
def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    if (lstLen % 2):
        return sortedLst[index]
    else:
        a = sortedLst[index] + sortedLst[index + 1]
        return (sortedLst[index] + sortedLst[index + 1])/2.0

ans_all = []
for i in range(len(com)):
    ans = []
    #名稱
    ans.append(name[i])  
    ans_all.append(ans)

    #幾筆資料
    ans.append(len(com[i][1:]))               
    #平均
    avr = float(sum(com[i][1:])/len(com[i][1:]))
    ans.append(avr)             
    #中位數
    med = median(com[i][1:])    
    ans.append(med)  


for i in range(len(ans_all)):
    print('%-10s' % ans_all[i][0],end="")
    print('%5s' % ans_all[i][1],end="")  
    print('%5.1f' % ans_all[i][2],end="")
    print('%5.1f' % ans_all[i][3])   