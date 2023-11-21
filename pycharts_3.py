from pyecharts import options as opts
from pyecharts.charts import Bar
import os
bar = Bar()
bar.add_xaxis(['diginetica','Nowplaying','Tmall'])
# P@20
bar.add_yaxis("Method1", [50.1430,18.9571,27.6122])
bar.add_yaxis("Method2", [54.1375,22.4806,32.6821])
# MRR@20
# bar.add_yaxis("Method1", [16.8209,7.2251,12.9747])
# bar.add_yaxis("Method2", [19.0455,8.5453,15.2775])
bar.set_global_opts(title_opts=opts.TitleOpts(title="两种体现用户长短期兴趣对比的方法",subtitle="指标为：P@20"))
bar.render("c.html")
os.system("c.html")
