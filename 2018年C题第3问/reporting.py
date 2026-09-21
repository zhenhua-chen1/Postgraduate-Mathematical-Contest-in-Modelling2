"""将已验证的分析表输出为中文图片与 Excel 快照，不改变 Spyder 图形后端。"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates, font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import PercentFormatter
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


BLUE, TEAL, ORANGE = '#245a81', '#248878', '#dd8b3a'
YEARS = [2015, 2016, 2017]
YEAR_COLORS = ['#adc7d6', '#5593b3', '#174d70']
LEVEL_COLORS = ['#ad3f3f', '#d27e4c', '#ddb55d', '#7bad9a', '#b9d5dc']


def _font():
    """优先系统中文字体；Windows、macOS、Linux 均有候选。"""
    for name in ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Microsoft YaHei',
                 'Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei']:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return 'DejaVu Sans'


def _figure(title, size=(13.6, 8.6)):
    fig = Figure(figsize=size, facecolor='white')
    # 独立 Agg 画布仅负责文件渲染；不调用 pyplot.show 或 matplotlib.use。
    FigureCanvasAgg(fig)
    fig.suptitle(title, x=.065, y=.965, ha='left', fontsize=19, fontweight='bold')
    fig.subplots_adjust(left=.085, right=.96, top=.86, bottom=.15,
                        hspace=.43, wspace=.36)
    return fig


def _axis(ax, title=None, ylabel=None, grid=True):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#bac5cc')
    ax.tick_params(colors='#394d5a', labelsize=9)
    if grid:
        ax.grid(axis='y', color='#e4eaee', linewidth=.7)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc='left', fontsize=12, pad=12, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel)


def _dates(ax, interval=6):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', labelrotation=25)


def _note(fig, text):
    fig.text(.065, .035, text, ha='left', va='bottom', fontsize=9,
             color='#566976', linespacing=1.65)


def _save(fig, folder, name, progress):
    path = folder / name
    fig.savefig(path, dpi=175, facecolor='white')
    fig.clear()
    progress(f'图片已保存：figures/{name}')
    return str(path)


def _monthly(tables):
    fig = _figure('近三年全球走势：次数、危害与季节性', (13.6, 10.4))
    fig.subplots_adjust(hspace=.56, bottom=.12, top=.88)
    axes = fig.subplots(3, 1)
    m = tables['全球月度历史'].sort_values('月份')
    for ax, col, color, title in [
        (axes[0], '事件数', BLUE, '事件数量：按真实年月重新计数'),
        (axes[1], '累计危害', TEAL, '累计危害：第一问统一回顾性评分的月度合计')]:
        ax.plot(m['月份'], m[col], color=color, lw=2.1, marker='o', ms=3)
        ax.fill_between(m['月份'], 0, m[col], color=color, alpha=.09)
        ax.set_ylim(bottom=0)
        _axis(ax, title, '起' if col == '事件数' else '得分合计')
        _dates(ax)
    for y, color in zip(YEARS, YEAR_COLORS):
        x = tables['季节性描述'].query('年份 == @y')
        axes[2].plot(x['月序号'], x['相对当年日均水平'], marker='o', color=color, label=str(y))
    axes[2].axhline(1, color='#8c9ca5', ls='--', lw=1)
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xlabel('月份')
    axes[2].legend(ncol=3, frameon=False, loc='upper right')
    _axis(axes[2], '按月天数及年度规模归一化：观察季节变化，不等于稳定季节规律', '当年日均 = 1')
    _note(fig, '描述窗口为2015—2017年；事件数量包含缺坐标事件。\n累计危害采用1998—2017年全样本回顾性标尺，仅用于描述，不参与预测及历史回测。')
    return fig


def _regions_levels(tables):
    fig = _figure('地区差异与危害级别分布')
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.22, 1]})
    p = tables['地区年度统计'].pivot(index='地区', columns='年份', values='事件数').fillna(0)
    p = p.loc[p.sum(axis=1).sort_values().index]
    y = np.arange(len(p))
    for offset, year, color in zip([-.24, 0, .24], YEARS, YEAR_COLORS):
        left.barh(y + offset, p[year], height=.22, color=color, label=str(year))
    left.set_yticks(y, p.index)
    left.legend(frameon=False, ncol=3, loc='lower right')
    left.set_xlabel('事件数（起）')
    _axis(left, '全部12个地区：按三年合计排序', grid=False)
    left.grid(axis='x', color='#e4eaee')
    left.set_axisbelow(True)
    level = tables['五级分布'].pivot(index='年份', columns='危害等级', values='年度占比').reindex(YEARS).fillna(0)
    bottom = np.zeros(3)
    for k, color in zip(range(1, 6), LEVEL_COLORS):
        v = level[k].values
        right.bar([str(x) for x in YEARS], v, bottom=bottom, color=color, label=f'{k}级')
        for j, val in enumerate(v):
            if val >= .065:
                right.text(j, bottom[j] + val / 2, f'{val:.1%}', ha='center', va='center',
                           fontsize=10, color='white' if k == 1 else '#263d4a')
        bottom += v
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.set_ylim(0, 1)
    right.legend(frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(.5, -.09))
    _axis(right, '五级占比：1级最严重，5级最轻', '当年事件占比')
    _note(fig, '地区以附件1的region字段为准。级别按eventid连接第一问评分，并核验事件集合与等级方向。\n级别分布采用统一回顾性标尺；等级编号本身不是可累加的危害强度。')
    return fig


def _land_polygons(base):
    path = base / 'assets' / 'ne_110m_land.geojson'
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding='utf-8'))
    polygons = []
    for item in obj.get('features', []):
        g = item.get('geometry') or {}
        coords = g.get('coordinates', [])
        parts = [coords] if g.get('type') == 'Polygon' else coords if g.get('type') == 'MultiPolygon' else []
        for p in parts:
            if p:
                ring = np.asarray(p[0], float)
                polygons.append(np.column_stack([ring[:, 0], np.sin(np.radians(ring[:, 1]))]))
    return polygons


def _spatial_maps(tables, base):
    fig = _figure('年度空间分布：等面积网格中的次数与累计危害', (15.2, 10.8))
    fig.subplots_adjust(left=.07, right=.965, top=.89, bottom=.20, hspace=.35, wspace=.13)
    axes = fig.subplots(3, 2)
    cells = tables['空间网格']
    evolution = tables['空间演变']
    nlon = int(evolution.iloc[0]['网格经度数'])
    nlat = int(evolution.iloc[0]['网格纬度数'])
    xedges, yedges = np.linspace(-180, 180, nlon + 1), np.linspace(-1, 1, nlat + 1)
    land = _land_polygons(base)
    handles = []
    for col, field, cmap in [(0, '事件数', 'YlOrRd'), (1, '累计危害', 'PuBuGn')]:
        positive = cells.loc[cells[field] > 0, field]
        low, high = float(positive.min()), float(positive.max())
        norm = LogNorm(vmin=low, vmax=max(high, low * 1.01))
        for row, year in enumerate(YEARS):
            ax = axes[row, col]
            for polygon in land:
                ax.add_patch(Polygon(polygon, closed=True, facecolor='#ebeeef', edgecolor='#c4cccf', lw=.25))
            block = cells[cells['年份'].eq(year)]
            z = np.zeros((nlat, nlon))
            z[block['纬度格'].to_numpy(int), block['经度格'].to_numpy(int)] = block[field]
            mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_less_equal(z, 0),
                                 norm=norm, cmap=cmap, shading='flat', rasterized=True)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-1, 1)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks(np.sin(np.radians([-60, 0, 60])), ['60°S', '0°', '60°N'])
            ax.tick_params(labelsize=8)
            ax.set_title(f'{year}年 · {field}', loc='left', fontsize=11, pad=6)
            for spine in ax.spines.values():
                spine.set_color('#d4dde2')
        handles.append(mesh)
    for c, handle in enumerate(handles):
        cax = fig.add_axes([.12 + c * .47, .133, .33, .016])
        cb = fig.colorbar(handle, cax=cax, orientation='horizontal')
        cb.set_label('事件数 / 格（对数色阶）' if c == 0 else '累计危害 / 格（对数色阶）', fontsize=9)
        cb.ax.tick_params(labelsize=8)
    _note(fig, '同列三年共用色阶；未观测到事件的网格透明。网格沿经度与sin(纬度)均分，每格面积一致。\n缺坐标事件不进入地图但保留其他统计；累计危害使用回顾性评分。地图变化只描述空间分布，不证明因果传播。\n底图：Natural Earth，公共领域。未提供底图文件时仍可绘制网格。')
    return fig


def _diffusion(tables):
    fig = _figure('空间变化：活跃范围、热点更替与国家事件数变化')
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.13]})
    e = tables['空间演变'].set_index('年份').reindex(YEARS)
    x = np.arange(3)
    for offset, field, label, color in [(-.25, '活跃格数', '活跃网格', BLUE),
                                        (0, '相对上年新增格数', '相对上年新增', TEAL),
                                        (.25, '相对上年消失格数', '相对上年消失', ORANGE)]:
        left.bar(x + offset, e[field], width=.23, label=label, color=color)
    left.set_xticks(x, [str(y) for y in YEARS])
    left.legend(frameon=False, fontsize=9, loc='upper center', ncol=3,
                bbox_to_anchor=(.5, -.065))
    _axis(left, '固定网格中的变化；2015年仅作基期', '网格数')
    p = tables['国家年度统计'].pivot(index='国家', columns='年份', values='事件数').fillna(0)
    change = (p[2017] - p[2015]).sort_values()
    selected = pd.concat([change.head(6), change.tail(6)]).loc[lambda s: ~s.index.duplicated()].sort_values()
    translate = {'Iraq': '伊拉克', 'Afghanistan': '阿富汗', 'Pakistan': '巴基斯坦', 'India': '印度',
                 'Nigeria': '尼日利亚', 'Syria': '叙利亚', 'Yemen': '也门', 'Philippines': '菲律宾',
                 'Ukraine': '乌克兰', 'Libya': '利比亚', 'Turkey': '土耳其', 'Somalia': '索马里',
                 'Egypt': '埃及', 'Burkina Faso': '布基纳法索', 'Mali': '马里', 'Kenya': '肯尼亚',
                 'Bangladesh': '孟加拉国', 'Nepal': '尼泊尔', 'United Kingdom': '英国',
                 'United States': '美国', 'Myanmar': '缅甸', 'Sri Lanka': '斯里兰卡',
                 'Democratic Republic of the Congo': '刚果（金）'}
    colors = [TEAL if val < 0 else ORANGE for val in selected]
    right.barh([translate.get(s, s) for s in selected.index], selected.values, color=colors)
    right.axvline(0, color='#657a86', lw=.8)
    right.set_xlabel('2017年减2015年（起）')
    _axis(right, '事件数增加及减少最多的各6个国家', grid=False)
    right.grid(axis='x', color='#e4eaee')
    right.set_axisbelow(True)
    _note(fig, '新增/消失指相对上一年的记录有无变化，并不等于首次发生或实际彻底消失。\n空间覆盖受地理编码、报告完整性和网格尺度影响；国家计数包含缺坐标事件。细、粗网格敏感性另见数据表。')
    return fig


def _holdout(tables):
    fig = _figure('2017独立留出回测：冻结模型与区间检验', (14.4, 9.8))
    fig.subplots_adjust(bottom=.15, hspace=.47)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    top = fig.add_subplot(grid[0, :])
    left, right = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    g = tables['2017逐月回测'].query('地区 == "全球"').sort_values('月份')
    top.fill_between(g['月份'], g['下限95'], g['上限95'], color=BLUE, alpha=.10, label='名义95%区间')
    top.fill_between(g['月份'], g['下限80'], g['上限80'], color=BLUE, alpha=.20, label='名义80%区间')
    top.plot(g['月份'], g['实际事件数'], color='#243b48', marker='o', lw=2, label='实际')
    top.plot(g['月份'], g['预测事件数'], color=ORANGE, lw=2, marker='o', label='冻结方案')
    top.legend(frameon=False, ncol=4, loc='upper right', fontsize=9)
    _axis(top, '全球：统一从2016年12月末预测未来12个月', '起')
    _dates(top, 1)
    metrics = tables['2017回测指标']
    selected = metrics.query('方案 == "冻结方案"').set_index('地区')
    baseline = metrics.query('方案 == "近12月均值基线"').set_index('地区')
    names = selected.drop(index='全球').sort_values('WAPE').index
    pos = np.arange(len(names))
    left.barh(pos - .18, selected.loc[names, 'WAPE'], .34, color=BLUE, label='冻结方案')
    left.barh(pos + .18, baseline.loc[names, 'WAPE'], .34, color='#b9c8d2', label='近12月均值')
    left.set_yticks(pos, names)
    left.xaxis.set_major_formatter(PercentFormatter(1))
    left.legend(frameon=False, loc='lower right', fontsize=9)
    _axis(left, '地区WAPE：越低越好，小样本地区波动大', grid=False)
    left.grid(axis='x', color='#e4eaee')
    left.set_axisbelow(True)
    right.scatter(selected.loc[names, '覆盖率80'], pos - .11, color=ORANGE, marker='o', s=32, label='名义80%')
    right.scatter(selected.loc[names, '覆盖率95'], pos + .11, color=TEAL, marker='s', s=30, label='名义95%')
    right.axvline(.8, ls='--', color=ORANGE, alpha=.55)
    right.axvline(.95, ls='--', color=TEAL, alpha=.55)
    right.set_yticks(pos, names)
    right.set_xlim(.45, 1.025)
    right.xaxis.set_major_formatter(PercentFormatter(1))
    right.legend(frameon=False, loc='lower left', fontsize=9)
    _axis(right, '实际月度覆盖率：虚线是名义水平', grid=False)
    right.grid(axis='x', color='#e4eaee')
    right.set_axisbelow(True)
    _note(fig, '模型仅用2012—2016年选定；2017年测试结果不参与换模型。并非所有地区都优于简单基线。\n中东与北非两档区间均仅覆盖8/12个月（66.7%）；小样本、结构变化与同选模期误差复用限制区间可靠性。')
    return fig


def _forecast(tables):
    fig = _figure('2018年月度研判：全球及近三年事件数最多的三个地区', (14.4, 9.8))
    fig.subplots_adjust(hspace=.47, bottom=.16)
    axes = fig.subplots(2, 2).ravel()
    order = tables['地区年度统计'].groupby('地区')['事件数'].sum().nlargest(3).index.tolist()
    monthly, regions = tables['全球月度历史'], tables['地区月度面板']
    for ax, name in zip(axes, ['全球', *order]):
        hist = monthly if name == '全球' else regions[regions['地区'].eq(name)]
        hist = hist.sort_values('月份')
        p = tables['2018月度预测'].query('地区 == @name').sort_values('月份')
        ax.plot(hist['月份'], hist['事件数'], color='#526979', lw=1.6, label='2015—2017实际')
        ax.fill_between(p['月份'], p['下限95'], p['上限95'], color=BLUE, alpha=.1, label='名义95%区间')
        ax.fill_between(p['月份'], p['下限80'], p['上限80'], color=BLUE, alpha=.22, label='名义80%区间')
        ax.plot(p['月份'], p['预测事件数'], color=ORANGE, lw=2, label='2018点预测')
        ax.axvline(pd.Timestamp('2018-01-01'), color='#7f929e', lw=.9, ls='--')
        ax.set_ylim(bottom=0)
        _axis(ax, name, '事件数（起）')
        _dates(ax, 12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(.5, .085))
    _note(fig, '点预测是非负期望次数，可为小数；全球等于全部12地区点预测加总。重点地区按2015—2017累计事件数选择。\n区间为历史误差RMS正态近似，名义水平不保证实际覆盖；地区或月份的区间端点不能直接加总。')
    return fig


def _annual_forecast(tables):
    fig = _figure('2018年度研判：与2017年比较及年度不确定性')
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [.8, 1.3]})
    annual = tables['2018年度预测'].set_index('地区')
    global_row = annual.loc['全球']
    left.bar([0, 1], [global_row['2017实际事件数'], global_row['预测事件数']],
             width=.55, color=['#9ab4c4', BLUE])
    left.errorbar(1, global_row['预测事件数'],
                  yerr=[[global_row['预测事件数'] - global_row['下限95']],
                        [global_row['上限95'] - global_row['预测事件数']]],
                  fmt='none', capsize=8, color=ORANGE, lw=2, label='名义95%年度区间')
    left.set_xticks([0, 1], ['2017实际', '2018预测'])
    left.set_ylim(0, global_row['上限95'] * 1.17)
    for i, val in enumerate([global_row['2017实际事件数'], global_row['预测事件数']]):
        left.text(i if i == 0 else i + .055, val + global_row['上限95'] * .025,
                  f'{val:,.0f}' if i == 0 else f'{val:,.1f}',
                  ha='center' if i == 0 else 'left', fontsize=11)
    left.legend(frameon=False, fontsize=9, loc='upper left')
    _axis(left, f'全球点预测同比 {global_row["相对2017变化率"]:+.1%}', '起')
    a = annual.drop(index='全球').sort_values('预测事件数')
    y = np.arange(len(a))
    right.errorbar(a['预测事件数'], y,
                   xerr=np.vstack([a['预测事件数'] - a['下限95'], a['上限95'] - a['预测事件数']]),
                   fmt='o', ms=4, color=BLUE, ecolor='#9ebccb', capsize=3, label='2018预测及名义95%区间')
    right.scatter(a['2017实际事件数'], y, color=ORANGE, marker='|', s=100, label='2017实际')
    right.set_yticks(y, a.index)
    right.set_xlim(left=0)
    right.set_xlabel('全年事件数（起）')
    right.legend(frameon=False, fontsize=9, loc='lower right')
    _axis(right, '地区年度区间：保留完整年度内误差相关性', grid=False)
    right.grid(axis='x', color='#e4eaee')
    right.set_axisbelow(True)
    _note(fig, '年度误差先在每个历史预测年内加总，再以6个完整年度误差估计RMS；未把月度区间端点相加。\n预测只反映截至2017年末的数据趋势，未使用2018年实际结果；区间很宽，应与地区情景和事件监测结合判断。')
    return fig


def _categories(tables):
    fig = _figure('事件特征结构：用于背景解释的描述性证据')
    fig.subplots_adjust(left=.135, wspace=.42)
    axes = fig.subplots(1, 2)
    translations = {'Bombing/Explosion': '爆炸/炸弹', 'Armed Assault': '武装袭击',
                    'Assassination': '暗杀', 'Hostage Taking (Kidnapping)': '绑架',
                    'Facility/Infrastructure Attack': '设施/基础设施袭击', 'Unknown': '未知',
                    'Unarmed Assault': '非武装袭击', 'Hijacking': '劫持',
                    'Hostage Taking (Barricade Incident)': '劫持人质/据守',
                    'Private Citizens & Property': '平民与私人财产', 'Military': '军队',
                    'Police': '警察', 'Government (General)': '政府（一般）',
                    'Business': '商业', 'Transportation': '交通运输',
                    'Religious Figures/Institutions': '宗教人员/机构',
                    'Educational Institution': '教育机构', 'Utilities': '公共事业',
                    'Terrorists/Non-State Militia': '恐怖组织/非国家武装',
                    'Violent Political Party': '暴力政治组织'}
    all_features = tables['事件特征统计']
    for ax, dimension in zip(axes, ['袭击类型', '目标类型']):
        p = all_features[all_features['维度'].eq(dimension)].pivot(index='类别', columns='年份', values='事件数').fillna(0)
        keep = p.sum(axis=1).nlargest(6).index
        ordered = p.loc[keep].sum(axis=1).sort_values().index
        shares = p.div(p.sum(axis=0), axis=1).loc[ordered]
        y = np.arange(len(shares))
        for offset, year, color in zip([-.24, 0, .24], YEARS, YEAR_COLORS):
            ax.barh(y + offset, shares[year], height=.22, color=color, label=str(year))
        ax.set_yticks(y, [translations.get(c, c) for c in shares.index])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, ncol=3, loc='lower right', fontsize=9)
        _axis(ax, f'{dimension}：三年合计数量前6类', grid=False)
        ax.grid(axis='x', color='#e4eaee')
        ax.set_axisbelow(True)
        ax.set_xlabel('占当年全部事件的比例')
    _note(fig, '仅统计附件中的第一袭击类型与第一目标类型；每个维度的分母均为当年全部事件数。\n类别占比是统计相关结构，不能单独识别恐怖袭击原因；归属记录和未知类别也不能当成司法确认。')
    return fig


def save_figures(tables: dict[str, pd.DataFrame], base: Path, output: Path,
                 progress=print) -> list[str]:
    """保存8张PNG，返回绝对路径；不弹窗，保持Spyder当前图形设置。"""
    folder = Path(output) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    tasks = [('01_全球走势与季节性.png', lambda: _monthly(tables)),
             ('02_地区与级别分布.png', lambda: _regions_levels(tables)),
             ('03_年度空间次数与危害.png', lambda: _spatial_maps(tables, Path(base))),
             ('04_空间演变与国家变化.png', lambda: _diffusion(tables)),
             ('05_2017独立回测.png', lambda: _holdout(tables)),
             ('06_2018月度预测.png', lambda: _forecast(tables)),
             ('07_2018年度预测.png', lambda: _annual_forecast(tables)),
             ('08_事件特征结构.png', lambda: _categories(tables))]
    paths = []
    with rc_context({'font.family': _font(), 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.labelcolor': '#394d5a',
                     'text.color': '#263d4a', 'savefig.bbox': None}):
        for number, (name, draw) in enumerate(tasks, 1):
            progress(f'绘图 {number}/{len(tasks)}：{name}')
            paths.append(_save(draw(), folder, name, progress))
    return paths


def _display_width(value):
    text = str(value) if value is not None else ''
    return max((sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in line)
                for line in text.splitlines()), default=0)


def export_workbook(tables, output, progress=print) -> Path:
    """一个主工作簿保留核心快照；候选模型的逐月明细由CSV提供。"""
    path = Path(output) / '2018年预测结果.xlsx'
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = ['2018年度预测', '2018月度预测', '2017回测指标', '2017逐月回测',
              '模型选择', '时间划分', '区间误差尺度', '年度概况', '全球月度历史',
              '地区年度统计', '五级分布', '空间演变', '网格敏感性',
              '数据核对', '原因与建议', '参考资料', '运行说明']
    with pd.ExcelWriter(path, engine='openpyxl', datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as writer:
        for name in sheets:
            if name not in tables:
                continue
            table = tables[name]
            table.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            ws.sheet_view.showGridLines = False
            ws.row_dimensions[1].height = 34
            ws.sheet_properties.pageSetUpPr.fitToPage = True
            ws.page_setup.orientation = 'landscape'
            ws.page_setup.paperSize = ws.PAPERSIZE_A4
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
            ws.print_title_rows = '1:1'
            for cell in ws[1]:
                cell.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF', size=10)
                cell.fill = PatternFill('solid', fgColor='245A81')
                cell.alignment = Alignment(vertical='center', wrap_text=True)
            for j, column in enumerate(table.columns, 1):
                text_col = table[column].dtype == object
                samples = table[column].dropna().head(200).tolist()
                width = min(65 if text_col else 27,
                            max(12, _display_width(column) + 2,
                                max((_display_width(v) + 2 for v in samples if isinstance(v, str)), default=0)))
                ws.column_dimensions[get_column_letter(j)].width = width
                percent = any(w in str(column) for w in ['占比', '覆盖率', '变化率', '同比', 'WAPE', '完整率'])
                integer = str(column) in ['年份', '预测年份', '步长', '预测步数', '危害等级', '预测季度',
                                         '历史完整年度数', '季度合并误差数', '网格经度数', '网格纬度数',
                                         '事件数', '实际事件数', '2017实际事件数', '一级事件数',
                                         '有效坐标数', '活跃格数', '相对上年新增格数', '相对上年消失格数',
                                         '已知死亡记录数', '已知受伤记录数', '已记录死亡人数', '已记录受伤人数',
                                         '空间事件数', '相邻年持续格数', '新增格事件数', '有事件国家数', '年度实际事件数']
                for row in range(2, ws.max_row + 1):
                    cell = ws.cell(row, j)
                    cell.font = Font(name='Microsoft YaHei', size=10, color='263D4A')
                    cell.alignment = Alignment(vertical='center', wrap_text=True)
                    if row % 2 == 0:
                        cell.fill = PatternFill('solid', fgColor='F0F5F8')
                    if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                        cell.number_format = '0.0%' if percent else ('0' if integer else '#,##0.00;[Red]-#,##0.00;0.00')
                        if str(column) in ['HHI', '平均危害', '相邻年Jaccard']:
                            cell.number_format = '0.0000'
                    elif isinstance(cell.value, (pd.Timestamp, np.datetime64)) or cell.is_date:
                        cell.number_format = 'yyyy-mm-dd'
            # 长解释和资料URL使用高一些的行，避免默认行高隐藏内容。
            for row in range(2, ws.max_row + 1):
                lines = max((int(np.ceil(_display_width(c.value) /
                                        max(1, ws.column_dimensions[get_column_letter(c.column)].width - 2)))
                             for c in ws[row]), default=1)
                ws.row_dimensions[row].height = min(135, max(24, 16 * lines + 8))
            progress(f'Excel工作表已写入：{name}（{len(table):,}行）')
    return path
