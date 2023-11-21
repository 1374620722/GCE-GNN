from pyecharts import options as opts
from pyecharts.charts import Bar
import os
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
# P@20
# bar.add_yaxis("Multi_scale", [54.1375,22.4806,32.6821])
# bar.add_yaxis("No_Multi_scale", [49.6697,20.2229,35.6475])
# MRR@20
bar.add_yaxis("Multi_scale", [19.0455,8.5453,15.2775])
bar.add_yaxis("No_Multi_scale", [17.3468,6.1758,16.0457])
bar.set_global_opts(title_opts=opts.TitleOpts(title="对比是否融合多尺度信息模型的性能",subtitle="指标为：MRR@20"))
bar.render("c.html")
os.system("c.html")
