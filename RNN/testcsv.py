import csv
f = open('/Users/qkrtm/Desktop/KAU/4-1/Jongsul/gitJongsul/RNN/dataset/종서박_glucose.csv','r', encoding='utf-8')
rdr = csv.reader(f)
cnt=1
for line in rdr:
    print(line[0])
    if('PM' in line[0]):
        print(line[0][13])
        if(int(line[0][13]) <12):
            print('small')
            rear = line[13:]
            rear = str(int(rear)+12)
            line[0]= line[0][0:13]
            line[0] = line[0]+ rear
    cnt=cnt+1
    print()
    if(cnt==3):
        break
f.close()
