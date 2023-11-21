from pyecharts import options as opts
from pyecharts.charts import Bar
import os
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
# P@20
# bar.add_yaxis("SR-GNN", [51.26,18.87,27.57])
# bar.add_yaxis("Our_model", [54.1375,22.4806,32.6821])
# MRR@20
bar.add_yaxis("SR-GNN", [17.78,7.47,13.72])
bar.add_yaxis("Our_model", [19.0455,8.5453,15.2775])
bar.set_global_opts(title_opts=opts.TitleOpts(title="对比经典模型SR-GNN",subtitle="指标为：MRR@20"))
bar.render("c.html")
os.system("c.html")
