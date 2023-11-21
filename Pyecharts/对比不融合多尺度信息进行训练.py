from pyecharts import options as opts
from pyecharts.charts import Bar
import os
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
# P@20
# bar.add_yaxis("未融合多尺度信息的模型", [49.6697,20.2229,35.6475])
# bar.add_yaxis("融合了多尺度信息后的模型", [54.0373,23.5338,33.5277])
# MRR@20
bar.add_yaxis("未融合多尺度信息的模型", [17.3468,6.1758,16.0457])
bar.add_yaxis("融合了多尺度信息后的模型", [19.0455,8.5453,15.2775])
bar.set_global_opts(title_opts=opts.TitleOpts(title="对比是否融合多尺度信息模型的性能",subtitle="指标为：MRR@20"))
bar.render("2.html")
os.system("2.html")