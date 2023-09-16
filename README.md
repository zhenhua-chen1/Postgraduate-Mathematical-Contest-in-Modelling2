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

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)、kmodes(0.12.10))和matplotlib(3.3.4)库；
      输入：附件2和3（无需操作）；
      功能：先用kmodes聚类后得到训练样本，再用该样本对支持向量机、随机森林、决策树训练模型，最后用准确度最高的模型给嫌疑人排序；
      输出：kmodes聚类个数变化图、预测准确度及未知案件嫌疑人排序表格。
      
3 2018年C题第三问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)、pmdarima(1.8.5和matplotlib(3.3.4)库；
      输入：问题二输出的文件（无需操作）；
      功能：用pm法预测2018年指标；
      输出：2018年结果图。
      
4 2019年D题第一问

      条件：numpy(1.20.1)、panadas(1.4.2)和scipy(1.6.2)库；
      输入：变量filenumber=1为输入文件1，文件2和3以此类推；
      功能：对时间序列、车速及怠速预处理；
      输出：result.xlsx文件。

5 2019年D题第二问

      条件：panadas(1.4.2)和matplotlib(3.3.4)库；
      输入：变量filenumber=1为输入文件1，文件2和3以此类推(该文件是第一问预处理完的)；
      功能：选取运动学片段；
      输出：运动学片段总数量、result.xlsx文件及某个片段图片。
      
6 2019年D题第三问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)和matplotlib(3.3.4)库；
      输入：变量filenumber=1为输入文件1，文件2和3以此类推(该文件是第二问所得运动学片段)；
      功能：运用k-means聚类分析把运动学片段分为三类并拼接成一条工矿曲线图；
      输出：运动学片段的9个指标文件、k-means聚类结果图、运动工矿曲线结果文件及图片和所得工矿曲线误差gap。

7 2020年B题第一问

      条件：numpy(1.20.1)和panadas(1.4.2)；
      输入：无需输入；
      功能：对样本285和313进行预处理；
      输出：285和313号样本预处理后的结果。
      
8 2021年B题第一问

      条件：numpy(1.20.1)和panadas(1.4.2)；
      输入：附件1；
      功能：根据题目给出的公式计算AQI；
      输出：8月25日-8月28日的AQI及首要污染物。
      
 9 2021年B题第二问

      条件：numpy(1.20.1)、panadas(1.4.2)和scikit-learn(0.24.1)；
      输入：附件1；
      功能：用KNN做数据预处理、用pearson系数分析相关性和用K-means作聚类分析；
      输出：各变量的划分后的类别及其相关性（请自己根据结果分析每一类的特征）。

 10 2021年B题第三问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)和matplotlib(3.3.4)库；
      输入：附件1和附近2；
      功能：用KNN做数据预处理和用随机森林进行预测；
      输出：预测图及A、B和C的13-15结果。
 
 11 2021年B题第四问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)和math库；
      输入：附件1和附近3；
      功能：用KNN做数据预处理和用随机森林进行预测；
      输出：A、A1、A2和A3的结果。

 12 2021年D题第一问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)和matplotlib(3.3.4)库；
      输入：ERα_activity和Molecular_Descriptor文件；
      功能：用3sigmal准则做数据预处理、用随机森林进行特征值分析并取前30个、用pearson计算相关性及剔除10个强相关性变量；
      输出：30个变量的特征图、相关性热力图和相关性文件及其最后的20个结果。
      
 13 2021年D题第二问

      条件：numpy(1.20.1)、panadas(1.4.2)、scikit-learn(0.24.1)和matplotlib(3.3.4)库；
      输入：ERα_activity和Molecular_Descriptor文件；
      功能：用KNN插值做、用线性回归、神经网络、随机森林和决策树进行回归预测；
      输出：各个预测拟合图和活性结果。
      
 14 2021年D题第三问

      条件：panadas(1.4.2)、scikit-learn(0.24.1)和matplotlib(3.3.4)库；
      输入：ERα_activity和ADMET文件；
      功能：用SVM、随机森林和决策树进行分类预测；
      输出：各个预测ROC图和分类结果文件。
 
 15 2021年D题第四问

      条件：numpy(1.20.1)、panadas(1.4.2)和scikit-learn(0.24.1)；
      输入：Molecular_Descriptor、ERα_activity和ADMET文件；
      功能：用遗传算法和随机森林优化范围；
      输出：20个变量的范围及5个ADMET性质。

 16 2022年E题第二问

      条件：numpy(1.20.1)、panadas(1.4.2)、keras(2.13.1)和scikit-learn(0.24.1)；
      输入：附近3、附近4和附近8；
      功能：使用LSTM、RNN、RF、ARIMA对时间序列进行预测；
      输出：真实值和预测值的结果对比图、水分和年份图、各土壤湿度表（问题二结果表）及各算法评价指标图。
      
