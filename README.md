# 数学建模数据分析代码
本人将不定时分享华为杯数学建模相关算法

能力有限，只上传会的代码

# 目前更新代码
1 2018年C题第一问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)、factor_analyzer(2.1.0)和matplotlib(3.3.4)库；
      输入：附件1（无需操作）；
      功能：运用PCA降维和k-means聚类分析把事件分类等级并提取十大恐怖事件；
      输出：降维后各均值变量文件、k-means聚类结果图、案件等级编号文件及危害程度最高的十大恐怖事件。

2 2018年C题第二问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)、kmodes(0.12.10)和matplotlib(3.3.4)库；
      输入：附件2和3（无需操作）；
      功能：在kmodes聚类后得到样本后，用该样本对支持向量机、随机森林、决策树训练模型，并最终使用准确度最高的模型给犯罪嫌疑人的可能性排序；
      输出：kmodes聚类个数变化图、预测准确度及未知案件嫌疑人排序表格。
      
3 2019年D题第一问

      条件：numpy(1.20.1)、panadas(1.4.2)和scipy(1.6.2)库；
      输入：变量filenumber=1为输入文件1，文件2和3以此类推；
      功能：对时间序列、车速及怠速预处理；
      输出：result.xlsx文件。

4 2019年D题第二问

      条件：panadas(1.4.2)和matplotlib(3.3.4)库；
      输入：变量filenumber=1为输入文件1，文件2和3以此类推(该文件是第一问预处理完的)；
      功能：选取运动学片段；
      输出：运动学片段总数量、result.xlsx文件及某个片段图片。
      
5 2019年D题第三问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)和matplotlib(3.3.4)库；
      输入：变量filenumber=1为输入文件1，文件2和3以此类推(该文件是第二问所得运动学片段)；
      功能：运用k-means聚类分析把运动学片段分为三类并拼接成一条工矿曲线图；
      输出：运动学片段的9个指标文件、k-means聚类结果图、运动工矿曲线结果文件及图片和所得工矿曲线误差gap。
