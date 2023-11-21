from pyecharts import options as opts
from pyecharts.charts import Bar
import os

attr = ['diginetica','Nowplaying','Tmall']

bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
bar.add_yaxis("全局和局部的项目转换信息+位置嵌入", [19.0455,8.5453,15.2775])
bar.add_yaxis("全局和局部的项目转换信息", [17.4086,6.5759,16.5610])
bar.add_yaxis("位置嵌入", [18.7945,7.5944,14.8448])
bar.set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
bar.render("b.html")
os.system("b.html")
