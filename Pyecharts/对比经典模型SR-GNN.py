from pyecharts import options as opts
from pyecharts.charts import Bar
import os
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
# P@20
# bar.add_yaxis("SR-GNN", [51.26,18.87,27.57])
# bar.add_yaxis("改进模型", [54.0373,23.5338,33.5277])
# MRR@20
bar.add_yaxis("SR-GNN", [17.78,7.47,13.72])
bar.add_yaxis("改进模型", [18.7785,7.6087,14.8648])
bar.set_global_opts(title_opts=opts.TitleOpts(title="对比经典模型SR-GNN",subtitle="指标为：MRR@20"))
bar.render("1.html")
os.system("1.html")
