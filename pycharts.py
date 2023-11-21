from pyecharts import options as opts
from pyecharts.charts import Bar
import os

attr = ['diginetica','Nowplaying','Tmall']

# P@20
# a = [54.1375,49.8455,53.9929]
# b = [22.4806,20.1639,23.2889]
# c = [32.6821,35.2267,33.2921]
# v1 = [54.1375,22.4806,32.6821]
# v2 = [49.8455,20.1639,35.2267]
# v3 = [53.9929,23.2889,33.2921]
# MRR@20
# a = [19.0455,17.4086,18.7945]
# b = [8.5453,6.5759,7.5944]
# c = [15.2775,16.5610,14.8448]
v1 = [19.0455,8.5453,15.2775]
v2 = [17.4086,6.5759,16.5610]
v3 = [18.7945,7.5944,14.8448]
# bar = Bar()
# bar.add("全局和局部的项目转换信息+位置嵌入", attr, v1, mark_line=['average'], mark_point=["max", "min"])  # 画平均线，标记最大最小值
# bar.add("全局和局部的项目转换信息", attr, v2, mark_line=['average'], mark_point=["max", "min"])
# bar.add("位置嵌入", attr, v3, mark_line=['average'], mark_point=["max", "min"])
#
# bar.render("a.html")
# os.system("a.html")
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
bar.add_yaxis("全局和局部的项目转换信息+位置嵌入", [19.0455,8.5453,15.2775])
bar.add_yaxis("全局和局部的项目转换信息", [17.4086,6.5759,16.5610])
bar.add_yaxis("位置嵌入", [18.7945,7.5944,14.8448])
bar.set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
bar.render("b.html")
os.system("b.html")
