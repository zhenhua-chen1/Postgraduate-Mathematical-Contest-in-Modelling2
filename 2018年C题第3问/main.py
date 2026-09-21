#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spyder 打开本文件，点击绿色箭头即可完整运行，无需参数或切换工作目录。

只读原附件，全部新结果写入本目录“优化结果”。描述2015—2017、预测2018，
不会随电脑日期改变。第一问评分只作回顾性描述，计数预测不使用该评分。
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Spyder 顶部工作目录可以位于别处，输入与输出都锚定到脚本位置。
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
try:
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from data_analysis import (read_events, validate_events, connect_scores, monthly_counts,
                               describe, spatial_analysis, sha256)
    from forecasting import run_forecasts
    from reporting import save_figures, export_workbook
    from conclusions import recommendations, write_analysis
except ImportError as exc:
    raise ImportError('请在当前Spyder解释器中安装requirements.txt中的依赖，并保留同目录辅助模块。'
                      f'原始错误：{exc}') from exc

OUTPUT = BASE / '优化结果'
SEED = 20260921


class Progress:
    """控制台实时刷新，同时把完整日志保存到结果目录。"""
    def __init__(self, output):
        self.start = perf_counter()
        self.path = output / '运行日志.txt'
        self.path.write_text('', encoding='utf-8')

    def __call__(self, message):
        line = f'[{datetime.now():%H:%M:%S} | {perf_counter() - self.start:7.1f}s] {message}'
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def summary_json(value):
    """JSON 中用 null 表示不适用的基线区间，不写非标准 NaN。"""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    return json.dumps(clean(value), ensure_ascii=False, indent=2,
                      default=json_default, allow_nan=False)


def check_results(tables):
    """验证实际计数、日期和加总关系，不能仅以保存文件成功作为验证。"""
    assert tables['年度概况'].set_index('年份')['事件数'].to_dict() == {2015: 14965, 2016: 13587, 2017: 10900}
    assert tables['全球月度历史']['事件数'].sum() == 39452
    assert tables['地区月度面板']['事件数'].sum() == 39452
    assert len(tables['地区月度面板']) == 36 * 12
    assert tables['五级分布']['事件数'].sum() == 39452
    assert tables['空间网格']['事件数'].sum() == 39244
    for key, year in [('2017逐月回测', 2017), ('2018月度预测', 2018)]:
        data = tables[key]
        for _, g in data.groupby('地区'):
            dates = pd.date_range(f'{year}-01-01', f'{year}-12-01', freq='MS')
            assert len(g) == 12 and np.array_equal(g['月份'].values, dates.values)
        values = data[['下限95', '下限80', '预测事件数', '上限80', '上限95']].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all()
        assert (np.diff(values, axis=1) >= -1e-9).all()
        total = data[data['地区'].eq('全球')].set_index('月份')['预测事件数']
        regions = data[~data['地区'].eq('全球')].groupby('月份')['预测事件数'].sum()
        assert np.allclose(total.values, regions.reindex(total.index).values, atol=1e-8)
    sums = tables['2018月度预测'].groupby('地区')['预测事件数'].sum()
    annual = tables['2018年度预测'].set_index('地区')['预测事件数']
    assert np.allclose(sums.sort_index(), annual.sort_index(), atol=1e-8)
    assert tables['时间划分'].iloc[-1]['训练结束'] == pd.Timestamp('2017-12-01')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log = Progress(OUTPUT)
    np.random.seed(SEED)
    summary = {'status': 'running', 'random_seed': SEED, 'information_cutoff': '2017-12-31'}
    summary_path = OUTPUT / 'run_summary.json'
    summary_path.write_text(summary_json(summary), encoding='utf-8')
    try:
        log('2018年C题第三问开始；输入只读，全部结果保存到优化结果')
        log(f'当前解释器：{sys.executable}')
        log('[1/8] 核对输入版本并读取原始附件')
        score_path = BASE.parent / '2018年C题第1问/优化结果/案件危害分级.csv'
        q1_input = BASE.parent / '2018年C题第1问/附件1.xlsx'
        # 只记录当前必需输入；统计均从原始事件重建，不再依赖旧版加工表。
        input_paths = [BASE / '附件1.xlsx', score_path, q1_input]
        hashes = {str(p.relative_to(BASE.parent)): sha256(p) for p in input_paths}
        if sha256(BASE / '附件1.xlsx') != sha256(q1_input):
            raise ValueError('第一问与第三问原始附件版本不同，不能直接连接评分。')
        raw = validate_events(read_events(BASE / '附件1.xlsx', log))
        log('[2/8] 连接回顾性评分，重建真实时间轴与地区面板')
        data = connect_scores(raw, score_path)
        recent = data[data.iyear.between(2015, 2017)]
        panel = monthly_counts(data)
        tables = describe(data)
        tables['历史计数面板'] = panel.reset_index()
        coordinate_counts = recent['坐标状态'].value_counts().to_dict()
        audit = [
            ('全部事件数', len(data), '原始附件1998—2017'),
            ('近三年事件数', len(recent), '按iyear选2015—2017，不解析eventid年份'),
            ('近三年有效坐标', coordinate_counts.get('有效', 0), '其他事件继续参加非空间统计'),
            ('近三年缺失坐标', coordinate_counts.get('缺失', 0), '不填(0,0)'),
            ('评分匹配事件数', len(data), '两附件SHA256一致，主键集合一致，连接一对一'),
            ('历史面板月份数', len(panel), '2007-01至2017-12，地区空月补0以全球完整月覆盖为前提'),
            ('历史面板地区数', panel.shape[1], '12地区×3年不是36个月'),
            ('全部事件编号年月与日期字段不同数', int(data.eventid.astype(str).str[:6].ne(data['month'].dt.strftime('%Y%m')).sum()), '编号不是日期权威字段'),
        ]
        tables['数据核对'] = pd.DataFrame(audit, columns=['核对项目', '结果', '说明'])
        log(f'近三年{len(recent):,}起，有效坐标{coordinate_counts.get("有效", 0):,}起，已从原始事件重建统计')
        log('[3/8] 计算年度空间格网、活跃格增减和分辨率敏感性')
        tables['空间网格'], tables['空间演变'] = spatial_analysis(data)
        tables['网格敏感性'] = pd.concat([spatial_analysis(data, *shape)[1]
                                           for shape in [(90, 45), (180, 90), (360, 180)]], ignore_index=True)
        log('[4/8] 滚动选模、2017完整年度留出与2018十二步预测')
        with threadpool_limits(limits=1):
            tables.update(run_forecasts(panel, log))
        log('[5/8] 校验预测日期、非负区间、地区加总与年度一致性')
        check_results(tables)
        tables['参考资料'] = pd.DataFrame(json.loads((BASE / 'sources.json').read_text(encoding='utf-8')))
        tables['原因与建议'] = recommendations(tables)
        tables['运行说明'] = pd.DataFrame([
            ('研究窗口', '2015—2017；预测2018；外部背景资料截至2017-12-31'),
            ('模型选择', '2012—2016五个完整年度，各截点最多60个月；2017留出不调参'),
            ('危害与级别', '第一问全样本回顾性评分仅作描述，一级最严重，不参加计数回测'),
            ('预测区间', '历史RMS×正态分位近似，月按步长季度合并，年按年度误差；不是保证覆盖'),
            ('总量一致', '12地区点预测加总为全球；区间端点不可跨月或跨地区直接相加'),
            ('缺失值', '伤亡缺失不填0；空月平均危害留空；缺坐标事件保留非空间统计'),
            ('工作簿性质', 'Python计算快照，改变输入后重跑main.py，不随Excel单元格自动重算'),
            ('详细数据', '候选逐月回测、误差尺度、网格明细另存csv；运行摘要记录来源和环境'),
        ], columns=['项目', '说明'])
        log('[6/8] 导出统一工作簿、CSV与分析报告')
        csvdir = OUTPUT / 'csv'
        csvdir.mkdir(exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(csvdir / f'{name}.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
        recent[['eventid', 'iyear', 'imonth', 'iday', '地区', 'country_txt', 'latitude', 'longitude',
                '坐标状态', 'specificity', '危害等级', '危害得分', '数据完整率']].to_csv(
                    csvdir / '近三年事件核验明细.csv', index=False, encoding='utf-8-sig')
        workbook = export_workbook(tables, OUTPUT, log)
        write_analysis(tables, OUTPUT)
        log('[7/8] 绘制时空、级别、回测和预测图')
        figures = save_figures(tables, BASE, OUTPUT, log)
        log('[8/8] 重新打开结果工作簿，与同次计算表逐项比较')
        with pd.ExcelFile(workbook) as xls:
            for name in xls.sheet_names:
                saved = pd.read_excel(xls, sheet_name=name)
                expected = tables[name]
                pd.testing.assert_frame_equal(saved, expected.reset_index(drop=True),
                                              check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-8)
        for p in input_paths:
            assert sha256(p) == hashes[str(p.relative_to(BASE.parent))], f'输入发生改变：{p.name}'
        global_forecast = tables['2018年度预测'].query('地区 == "全球"').iloc[0].to_dict()
        summary.update(status='completed', elapsed_seconds=round(perf_counter() - log.start, 2),
                       run_at=datetime.now().astimezone().isoformat(), executable=sys.executable,
                       python=platform.python_version(), platform=platform.platform(),
                       dependencies={p: importlib.metadata.version(p) for p in
                                     ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'openpyxl', 'threadpoolctl']},
                       input_sha256=hashes, code_sha256={p.name: sha256(p) for p in BASE.glob('*.py')},
                       sources_sha256=sha256(BASE / 'sources.json'),
                       events=len(data), recent_events=len(recent), coordinate_counts=coordinate_counts,
                       global_2018=global_forecast,
                       global_2017_metrics=tables['2017回测指标'].query('地区 == "全球"').to_dict('records'),
                       selected_models=tables['模型选择'].query('入选')[['地区', '模型']].to_dict('records'),
                       fitting_exceptions=len(tables['拟合异常']), figures=figures,
                       verification='日期、计数、评分、区间、加总、Excel重读及输入未改全部通过',
                       limitations=['事件时间回测，不是历史数据版本回测', '评分仅回顾性描述',
                                    '区间复用选模期误差，有乐观偏差；正态和稳定性为假设',
                                    '独立测试只有一年，部分地区不如基线且区间覆盖不足',
                                    '2012采集变化；空间重分布不等于因果传播；未验证2018真值'])
        summary_path.write_text(summary_json(summary), encoding='utf-8')
        log(f'完成：全球2018点预测{global_forecast["预测事件数"]:,.1f}起。结果目录：{OUTPUT}')
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        summary_path.write_text(summary_json(summary), encoding='utf-8')
        log(f'运行失败：{exc}')
        raise


if __name__ == '__main__':
    main()
