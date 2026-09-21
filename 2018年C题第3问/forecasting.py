"""冻结方案后的十二步年度回测、非负预测、总量协调与经验预测区间。"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

MODELS = ['季节朴素', '近12月均值', '阻尼季节ETS', '低阶SARIMA']
VALIDATION_YEARS = tuple(range(2012, 2017))
WINDOW = 60


def predict(y, model: str, horizon=12) -> np.ndarray:
    """同一截点预测后续十二个月，不使用预测期间的任何观测值。"""
    # 先验证模型名称，常量序列的快速返回也不能接受未知模型。
    if model not in MODELS:
        raise ValueError(f'未知模型：{model}')
    a = np.asarray(y, dtype=float)[-WINDOW:]
    if len(a) < 24 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('预测输入需要至少24个连续月份的非负观测。')
    if model == '季节朴素':
        pred = np.resize(a[-12:], horizon)
    elif model == '近12月均值':
        pred = np.repeat(a[-12:].mean(), horizon)
    elif np.ptp(a) == 0:
        pred = np.repeat(a[-1], horizon)
    else:
        with warnings.catch_warnings():
            # 优化不收敛时该次拟合失败，不能把警告吞掉后声称模型有效。
            warnings.simplefilter('error', ConvergenceWarning)
            # statsmodels 会自动将不可逆的“初始猜测”改为零；这不是最终收敛失败。
            warnings.filterwarnings('ignore', message='Non-invertible starting.*', category=UserWarning)
            if model == '阻尼季节ETS':
                fit = ExponentialSmoothing(a, trend='add', damped_trend=True,
                                           seasonal='add', seasonal_periods=12,
                                           initialization_method='estimated').fit(optimized=True)
            elif model == '低阶SARIMA':
                fit = SARIMAX(a, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                              trend='n', enforce_stationarity=True,
                              enforce_invertibility=True).fit(disp=False, maxiter=200)
                if not fit.mle_retvals.get('converged', False):
                    raise ValueError('SARIMA未收敛')
            else:
                raise ValueError(f'未知模型：{model}')
            pred = np.asarray(fit.forecast(horizon))
    if not np.isfinite(pred).all():
        raise ValueError('模型产生非有限预测值')
    return np.maximum(pred, 0.0)


def metrics(actual, pred) -> dict:
    y, p = np.asarray(actual), np.asarray(pred)
    error = p - y
    return {'MAE': float(np.mean(np.abs(error))),
            'RMSE': float(np.sqrt(np.mean(error ** 2))),
            'WAPE': float(np.abs(error).sum() / y.sum()) if y.sum() else np.nan,
            '偏差_预测减实际': float(np.mean(error))}


def reconcile(regional, global_pred, shares):
    """按初始地区预测占比协调，总量为零时使用训练期末十二个月占比。"""
    regional = np.asarray(regional, float)
    den = regional.sum(axis=1, keepdims=True)
    weights = np.divide(regional, den, out=np.tile(shares, (len(regional), 1)), where=den > 0)
    return weights * np.asarray(global_pred)[:, None]


def interval(pred, errors, annual=False):
    """以历史完整年度预测误差估计尺度。区间是近似值，不承诺名义覆盖率。

    月度按预测步长所属季度合并误差；年度先逐年加总误差，保留月际相关性。
    不可将十二个月区间端点相加当成年区间。
    """
    e = np.asarray(errors, float)  # 年份×十二个月
    p = np.asarray(pred, float)
    if e.ndim != 2 or e.shape[1] != 12 or not np.isfinite(e).all():
        raise ValueError('校准误差必须为完整的 年份×12 矩阵')
    if annual:
        sigma = np.sqrt(np.mean(e.sum(axis=1) ** 2))
    else:
        rms = np.sqrt(np.mean(e.reshape(-1, 4, 3) ** 2, axis=(0, 2)))
        sigma = np.repeat(rms, 3)
    out = {}
    for coverage in [80, 95]:
        z = norm.ppf(.5 + coverage / 200)
        out[f'下限{coverage}'] = np.maximum(0, p - z * sigma)
        out[f'上限{coverage}'] = p + z * sigma
    return out


def run_forecasts(panel: pd.DataFrame, progress=print) -> dict[str, pd.DataFrame]:
    """先完成2012—2016选择，再生成2017留出结果，最终方案不依据2017改选。"""
    series = dict(panel.items())
    series['全球'] = panel.sum(axis=1)
    cv, comparison, fit_log, splits = {}, [], [], []
    for year in [*VALIDATION_YEARS, 2017, 2018]:
        end = pd.Timestamp(year - 1, 12, 1)
        splits.append({'预测年份': year, '训练开始': end - pd.DateOffset(months=WINDOW - 1),
                       '训练结束': end, '预测开始': pd.Timestamp(year, 1, 1),
                       '预测结束': pd.Timestamp(year, 12, 1), '预测步数': 12,
                       '用途': '选模与区间估计' if year < 2017 else ('独立留出测试' if year == 2017 else '最终预测')})
    for j, (name, y) in enumerate(series.items(), 1):
        progress(f'模型比较 {j}/{len(series)}：{name}，5次完整年度回测 × 4个模型')
        for model in MODELS:
            predictions = []
            for year in VALIDATION_YEARS:
                train = y.loc[:f'{year - 1}-12-01']
                try:
                    predictions.append(predict(train, model))
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
                    fit_log.append({'地区': name, '模型': model, '预测年份': year,
                                    '处理': '候选排除', '原因': str(exc)[:250]})
                    predictions.append(np.full(12, np.nan))
            pred = np.asarray(predictions)
            actual = np.asarray([y.loc[f'{yr}-01-01':f'{yr}-12-01'].values for yr in VALIDATION_YEARS])
            cv[name, model] = pred
            ok = bool(np.isfinite(pred).all())
            comparison.append({'地区': name, '模型': model, '有效': ok,
                               **(metrics(actual, pred) if ok else {k: np.nan for k in metrics(actual, actual)})})
    cmp = pd.DataFrame(comparison)
    # 同分按预设候选顺序优先简单模型，2017数据不参与选择。
    winners = {}
    for name in panel.columns:
        rows = cmp[(cmp['地区'] == name) & cmp['有效']]
        winners[name] = rows.sort_values('MAE', kind='stable').iloc[0]['模型']
    bottomup = sum(cv[name, winners[name]] for name in panel.columns)
    global_actual = np.array([series['全球'].loc[f'{yr}-01-01':f'{yr}-12-01'].values for yr in VALIDATION_YEARS])
    cmp = pd.concat([cmp, pd.DataFrame([{'地区': '全球', '模型': '地区加总', '有效': True,
                                       **metrics(global_actual, bottomup)}])], ignore_index=True)
    global_model = cmp[(cmp['地区'] == '全球') & cmp['有效']].sort_values('MAE', kind='stable').iloc[0]['模型']
    winners['全球'] = global_model
    cmp['入选'] = [winners[n] == m for n, m in zip(cmp['地区'], cmp['模型'])]
    progress(f'方案已冻结：全球使用{global_model}；接下来仅评估2017，不据此换模型')
    for name, m in winners.items():
        progress(f'  {name}：{m}')

    def known_shares(year):
        totals = panel.loc[f'{year - 1}-01-01':f'{year - 1}-12-01'].sum().values.astype(float)
        return totals / totals.sum()

    val_reg = []
    for k, yr in enumerate(VALIDATION_YEARS):
        r = np.column_stack([cv[n, winners[n]][k] for n in panel.columns])
        g = r.sum(axis=1) if global_model == '地区加总' else cv['全球', global_model][k]
        val_reg.append(reconcile(r, g, known_shares(yr)))
    val_reg = np.asarray(val_reg)
    validation = np.concatenate([val_reg.sum(axis=2, keepdims=True), val_reg], axis=2)
    names = ['全球', *panel.columns]
    truth = np.asarray([np.column_stack([series[n].loc[f'{yr}-01-01':f'{yr}-12-01'] for n in names])
                        for yr in VALIDATION_YEARS])

    def future(year):
        out = {}
        for name in series:
            if name == '全球' and global_model == '地区加总':
                continue
            model = winners[name]
            try:
                out[name] = predict(series[name].loc[:f'{year - 1}-12-01'], model)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
                fit_log.append({'地区': name, '模型': model, '预测年份': year,
                                '处理': '明确回退季节朴素', '原因': str(exc)[:250]})
                progress(f'警告：{name} {year} 拟合失败，按预先约定回退季节朴素')
                out[name] = predict(series[name].loc[:f'{year - 1}-12-01'], '季节朴素')
        r = np.column_stack([out[n] for n in panel.columns])
        g = r.sum(axis=1) if global_model == '地区加总' else out['全球']
        r = reconcile(r, g, known_shares(year))
        return np.column_stack([r.sum(axis=1), r])

    test = future(2017)
    actual_test = np.column_stack([series[n].loc['2017-01-01':'2017-12-01'] for n in names])
    final = future(2018)
    errors = truth - validation
    monthly, annual, test_rows, test_metrics, calibration = [], [], [], [], []
    val_rows, candidate_rows = [], []
    for n, name in enumerate(names):
        e = errors[:, :, n]
        bounds_test = interval(test[:, n], e)
        actual_2017 = float(actual_test[:, n].sum())
        predicted_2017 = float(test[:, n].sum())
        # 年度留出区间只使用2012—2016完整年度误差，保留月际相关性。
        annual_bounds_test = interval(predicted_2017, e, annual=True)
        # 完成独立测试之后，2017的残差可加入2018区间估计；不改变点预测模型。
        final_e = np.vstack([e, actual_test[:, n] - test[:, n]])
        bounds_final = interval(final[:, n], final_e)
        for label, pred in [('冻结方案', test[:, n]),
                            ('季节朴素基线', predict(series[name].loc[:'2016-12-01'], '季节朴素')),
                            ('近12月均值基线', predict(series[name].loc[:'2016-12-01'], '近12月均值'))]:
            stats = {'地区': name, '方案': label, **metrics(actual_test[:, n], pred)}
            if label == '冻结方案':
                stats.update({'年度实际事件数': actual_2017, '年度预测事件数': predicted_2017,
                              '年度误差_预测减实际': predicted_2017 - actual_2017})
                for cov in [80, 95]:
                    lo, hi = bounds_test[f'下限{cov}'], bounds_test[f'上限{cov}']
                    stats[f'覆盖率{cov}'] = np.mean((actual_test[:, n] >= lo) & (actual_test[:, n] <= hi))
                    stats[f'平均宽度{cov}'] = np.mean(hi - lo)
                    annual_lo = float(annual_bounds_test[f'下限{cov}'])
                    annual_hi = float(annual_bounds_test[f'上限{cov}'])
                    stats[f'年度下限{cov}'] = annual_lo
                    stats[f'年度上限{cov}'] = annual_hi
                    stats[f'年度覆盖{cov}'] = bool(annual_lo <= actual_2017 <= annual_hi)
            test_metrics.append(stats)
        for h in range(12):
            common = {'地区': name, '地区初始模型': winners[name], '全球协调模型': global_model, '步长': h + 1}
            monthly.append({**common, '月份': pd.Timestamp(2018, h + 1, 1),
                            '预测事件数': final[h, n], **{k: v[h] for k, v in bounds_final.items()}})
            test_rows.append({**common, '月份': pd.Timestamp(2017, h + 1, 1),
                              '实际事件数': actual_test[h, n], '预测事件数': test[h, n],
                              **{k: v[h] for k, v in bounds_test.items()}})
        for k, yr in enumerate(VALIDATION_YEARS):
            for h in range(12):
                val_rows.append({'地区': name, '月份': pd.Timestamp(yr, h + 1, 1),
                                 '实际事件数': truth[k, h, n], '协调预测事件数': validation[k, h, n],
                                 '误差_实际减预测': e[k, h]})
        ay = float(final[:, n].sum())
        row = {'地区': name, '年份': 2018, '2017实际事件数': actual_2017,
               '预测事件数': ay, '相对2017变化率': ay / actual_2017 - 1 if actual_2017 else np.nan,
               **interval(ay, final_e, annual=True)}
        annual.append(row)
        for stage, err in [('2017测试', e), ('2018预测', final_e)]:
            for quarter in range(4):
                calibration.append({'地区': name, '用途': stage, '预测季度': quarter + 1,
                                    '历史完整年度数': len(err), '季度合并误差数': len(err) * 3,
                                    '误差RMS': float(np.sqrt(np.mean(err[:, quarter * 3:(quarter + 1) * 3] ** 2)))})
    for (name, model), pred in cv.items():
        for k, yr in enumerate(VALIDATION_YEARS):
            y = series[name].loc[f'{yr}-01-01':f'{yr}-12-01'].values
            for h in range(12):
                candidate_rows.append({'地区': name, '模型': model, '月份': pd.Timestamp(yr, h + 1, 1),
                                       '实际事件数': y[h], '预测事件数': pred[k, h]})
    return {'2018月度预测': pd.DataFrame(monthly), '2018年度预测': pd.DataFrame(annual),
            '模型选择': cmp, '2017回测指标': pd.DataFrame(test_metrics),
            '2017逐月回测': pd.DataFrame(test_rows), '选模期协调预测': pd.DataFrame(val_rows),
            '候选逐月回测': pd.DataFrame(candidate_rows), '时间划分': pd.DataFrame(splits),
            '区间误差尺度': pd.DataFrame(calibration),
            '拟合异常': pd.DataFrame(fit_log, columns=['地区', '模型', '预测年份', '处理', '原因'])}
