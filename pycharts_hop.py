from pyecharts import options as opts
from pyecharts.charts import Bar
import os
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
# P@20
# bar.add_yaxis("1-hop", [53.6363,22.4806,32.6821])
# bar.add_yaxis("2-hop", [54.1375,22.5073,22.5073])
# MRR@20
bar.add_yaxis("1-hop", [19.0606,8.5453,15.2775])
bar.add_yaxis("2-hop", [19.0455,8.4958,15.2293])
bar.set_global_opts(title_opts=opts.TitleOpts(title="全局图的不同跳数对比",subtitle="指标为：MRR@20"))
bar.render("c.html")
os.system("c.html")
