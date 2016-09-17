# import itertools
#
# class Apriori:
#     def __init__(self, min_sup=0.2, dataDic={}):
#         self.data = dataDic
#         self.size = len(dataDic) #Get the number of events
#         self.min_sup = min_sup
#         self.min_sup_val = min_sup * self.size
#
#     def find_frequent_1_itemsets(self):
#         FreqDic = {}  #{itemset1:freq1,itemsets2:freq2}
#         for event in self.data:
#             for item in self.data[event]:
#                 if item in FreqDic:
#                     FreqDic[item] += 1
#                 else:
#                     FreqDic[item] = 1
#         L1 = []
#         for itemset in FreqDic:
#             if itemset >= self.min_sup_val:
#                 L1.append([itemset])
#         return L1
#
#     def has_infrequent_subset(self,c,L_last,k):
#         subsets = list(itertools.combinations(c,k-1)) #return list of tuples of items
#         for each in subsets:
#             each = list(each) #change tuple into list
#             if each not in L_last:
#                 return True
#         return False
#
#     def apriori_gen(self,L_last): #L_last means frequent(k-1) itemsets
#         k = len(L_last[1]) + 1
#         Ck = []
#         for itemset1 in L_last:
#             for itemset2 in L_last:
#                 #join step
#                 flag = 0
#                 for i in range(k-2):
#                     if itemset1[i] != itemset2[i]:
#                         flag = 1 #the two itemset can't join
#                         break;
#                 if flag == 1:continue
#                 if itemset1[k-2] < itemset2[k-2]:
#                     c = itemset1 + [itemset2[k-2]]
#                 else:
#                     continue
#
#                 #pruning setp
#                 if self.has_infrequent_subset(c,L_last,k):
#                     continue
#                 else:
#                     Ck.append(c)
#         return Ck
#
#     def do(self):
#         L_last = self.find_frequent_1_itemsets()
#         L = L_last
#         i = 0
#         while L_last != []:
#             Ck = self.apriori_gen(L_last)
#             FreqDic = {}
#             for event in self.data:
#                 #get all suported subsets
#                 for c in Ck:
#                     if set(c) <= set(self.data[event]):#is subset
#                         if tuple(c) in FreqDic:
#                             FreqDic[tuple(c)]+=1
#                         else:
#                             FreqDic[tuple(c)]=1
#             print (FreqDic)
#             Lk = []
#             for c in FreqDic:
#                 if FreqDic[c] > self.min_sup_val:
#                     Lk.append(list(c))
#             L_last = Lk
#             L += Lk
#         return L


##________________________________________________________________________________________________________
# # -*- coding: utf-8 -*-
# """
# Apriori exercise.
# Created on Fri Nov 27 11:09:03 2015
# @author: 90Zeng
# """
# def loadDataSet():
#   '''创建一个用于测试的简单的数据集'''
#   return [ [ 1, 3, 4 ], [ 2, 3, 5 ], [ 1, 2, 3, 5 ], [ 2, 5 ] ]
# def createC1( dataSet ):
#   '''
#   构建初始候选项集的列表，即所有候选项集只包含一个元素，
#   C1是大小为1的所有候选项集的集合
#   '''
#   C1 = []
#   for transaction in dataSet:
#     for item in transaction:
#       if [ item ] not in C1:
#         C1.append( [ item ] )
#   C1.sort()
#   return map( frozenset, C1 )
# def scanD( D, Ck, minSupport ):
#   '''
#   计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
#   返回满足最小支持度的项集的集合，和所有项集支持度信息的字典。
#   '''
#   ssCnt = {}
#   for tid in D:
#     # 对于每一条transaction
#     for can in Ck:
#       # 对于每一个候选项集can，检查是否是transaction的一部分
#       # 即该候选can是否得到transaction的支持
#       if can.issubset( tid ):
#         ssCnt[ can ] = ssCnt.get( can, 0) + 1
#   numItems = float(len(D))
#   retList = []
#   supportData = {}
#   for key in ssCnt:
#     # 每个项集的支持度
#     support = ssCnt[key] / numItems
#     # 将满足最小支持度的项集，加入retList
#     if support >= minSupport:
#       retList.insert( 0, key )
#     # 汇总支持度数据
#     supportData[ key ] = support
#   return retList, supportData
#
# if __name__ == "__main__":
#     # 导入数据集
#     myDat = loadDataSet()
#     # 构建第一个候选项集列表C1
#     C1 = createC1(myDat)
#
#     # 构建集合表示的数据集 D
#     D = map(set, myDat)
#     # 选择出支持度不小于0.5 的项集作为频繁项集
#     L, suppData = scanD(D, C1, 0.5)
#
#     print (u"频繁项集L：", L)
#     print (u"所有候选项集的支持度信息：", suppData)

##______________________________________________________________________________________________________________

#-*- encoding: UTF-8 -*-
#---------------------------------import------------------------------------
class Apriori(object):

    def __init__(self, filename, min_support, item_start, item_end):
        self.filename = filename
        self.min_support = min_support # 最小支持度
        self.min_confidence = 50
        self.line_num = 0 # item的行数
        self.item_start = item_start #  取哪行的item
        self.item_end = item_end

        self.location = [[i] for i in range(self.item_end - self.item_start + 1)]
        self.support = self.sut(self.location)
        self.num = list(sorted(set([j for i in self.location for j in i])))# 记录item

        self.pre_support = [] # 保存前一个support,location,num
        self.pre_location = []
        self.pre_num = []

        self.item_name = [] # 项目名
        self.find_item_name()
        self.loop()
        self.confidence_sup()

    def deal_line(self, line):
        "提取出需要的项"
        return [i.strip() for i in line.split(' ') if i][self.item_start - 1:self.item_end]

    def find_item_name(self):
        "根据第一行抽取item_name"
        with open(self.filename, 'r') as F:
            for index,line in enumerate(F.readlines()):
                if index == 0:
                    self.item_name = self.deal_line(line)
                    break

    def sut(self, location):
        """
        输入[[1,2,3],[2,3,4],[1,3,5]...]
        输出每个位置集的support [123,435,234...]
        """
        with open(self.filename, 'r') as F:
            support = [0] * len(location)
            for index,line in enumerate(F.readlines()):
                if index == 0: continue
                # 提取每信息
                item_line = self.deal_line(line)
                for index_num,i in enumerate(location):
                    flag = 0
                    for j in i:
                        if item_line[j] != 'T':
                            flag = 1
                            break
                    if not flag:
                        support[index_num] += 1
            self.line_num = index # 一共多少行,出去第一行的item_name
        return support

    def select(self, c):
        "返回位置"
        stack = []
        for i in self.location:
            for j in self.num:
                if j in i:
                    if len(i) == c:
                        stack.append(i)
                else:
                    stack.append([j] + i)
        # 多重列表去重
        import itertools
        s = sorted([sorted(i) for i in stack])
        location = list(s for s,_ in itertools.groupby(s))
        return location

    def del_location(self, support, location):
        "清除不满足条件的候选集"
        # 小于最小支持度的剔除
        for index,i in enumerate(support):
            if i < self.line_num * self.min_support / 100:
                support[index] = 0
        # apriori第二条规则,剔除
        for index,j in enumerate(location):
            sub_location = [j[:index_loc] + j[index_loc+1:]for index_loc in range(len(j))]
            flag = 0
            for k in sub_location:
                if k not in self.location:
                    flag = 1
                    break
            if flag:
                support[index] = 0
        # 删除没用的位置
        location = [i for i,j in zip(location,support) if j != 0]
        support = [i for i in support if i != 0]
        return support, location

    def loop(self):
        "s级频繁项级的迭代"
        s = 2
        while True:
            print ('-'*80)
            print ('The' ,s - 1,'loop')
            print ('location' , self.location)
            print ('support' , self.support)
            print ('num' , self.num)
            print ('-'*80)

            # 生成下一级候选集
            location = self.select(s)
            support = self.sut(location)
            support, location = self.del_location(support, location)
            num = list(sorted(set([j for i in location for j in i])))
            s += 1
            if  location and support and num:
                self.pre_num = self.num
                self.pre_location = self.location
                self.pre_support = self.support

                self.num = num
                self.location = location
                self.support = support
            else:
                break

    def confidence_sup(self):
        "计算confidence"
        if sum(self.pre_support) == 0:
            print ('min_support error') # 第一次迭代即失败
        else:
            for index_location,each_location in enumerate(self.location):
                del_num = [each_location[:index] + each_location[index+1:] for index in range(len(each_location))] # 生成上一级频繁项级
                del_num = [i for i in del_num if i in self.pre_location] # 删除不存在上一级频繁项级子集
                del_support = [self.pre_support[self.pre_location.index(i)] for i in del_num if i in self.pre_location] # 从上一级支持度查找
                # print del_num
                # print self.support[index_location]
                # print del_support
                for index,i in enumerate(del_num): # 计算每个关联规则支持度和自信度
                    index_support = 0
                    if len(self.support) != 1:
                        index_support = index
                    support =  float(self.support[index_location])/self.line_num * 100 # 支持度
                    s = [j for index_item,j in enumerate(self.item_name) if index_item in i]
                    if del_support[index]:
                        confidence = float(self.support[index_location])/del_support[index] * 100
                        if confidence > self.min_confidence:
                            print (','.join(s) , '->>' , self.item_name[each_location[index]] , ' min_support: ' ,
                                   str(support) + '%' , ' min_confidence:' , str(confidence) + '%')

# def main():
    # c = Apriori('basket.txt', 14, 3, 13)
    # d = Apriori('simple.txt', 50, 2, 6)

if __name__ == '__main__':
    a = Apriori('./data/question1_0.txt', 10, 1, 1001)

