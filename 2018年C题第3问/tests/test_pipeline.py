"""第三问关键统计和预测约束的回归测试；不读取或改写第一问文件。

所有小样本均为合成数据。集成测试替换昂贵的拟合器，检查时间隔离、
模型冻结、结果结构与总量协调；真实模型精度由 main.py 的完整回测报告。
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

# 从任意工作目录执行测试，均使用本题目录中的模块。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_analysis as data
import forecasting as forecast


def event_frame(**overrides):
    """足够通过原始数据校验的一行样本，可按需替换字段。"""
    row = dict(eventid=201701010001, iyear=2017, imonth=1, iday=1,
               region=1, latitude=22.5, longitude=114.1,
               nkill=0, nwound=0, nkillter=0, nwoundte=0)
    row.update(overrides)
    return pd.DataFrame([row])


def seasonal_panel():
    """每地区具有稳定年度季节性，2017 年是完全独立的留出段。"""
    dates = pd.date_range('2007-01-01', '2017-12-01', freq='MS')
    return pd.DataFrame({name: dates.month.to_numpy() + 10 * code
                         for code, name in data.REGIONS.items()}, index=dates)


def cheap_predict(y, model, horizon=12):
    """可预知优胜模型的廉价拟合替身，保留真实基线行为。"""
    values = np.asarray(y, dtype=float)
    if model == '近12月均值':
        return np.repeat(values[-12:].mean(), horizon)
    penalty = {'季节朴素': 0, '阻尼季节ETS': 25, '低阶SARIMA': 50}[model]
    return np.resize(values[-12:], horizon) + penalty


class EventValidationTests(unittest.TestCase):
    def test_month_uses_date_columns_instead_of_eventid_prefix(self):
        d = data.validate_events(event_frame(eventid=201603150999, iyear=2017,
                                            imonth=2, iday=0))
        self.assertEqual(d.loc[0, 'month'], pd.Timestamp('2017-02-01'))
        self.assertEqual(d.loc[0, 'iday'], 0)  # 日未知不妨碍月度计数。

    def test_validation_does_not_modify_input(self):
        original = event_frame(nkill=-99)
        expected = original.copy(deep=True)
        data.validate_events(original)
        pd.testing.assert_frame_equal(original, expected)

    def test_duplicate_or_missing_identifiers_are_rejected(self):
        for sample in [pd.concat([event_frame(), event_frame()], ignore_index=True),
                       event_frame(eventid=np.nan)]:
            with self.subTest(sample=sample.eventid.tolist()):
                with self.assertRaisesRegex(ValueError, '编号'):
                    data.validate_events(sample)

    def test_invalid_calendar_month_year_and_region_are_rejected(self):
        cases = [dict(imonth=0), dict(imonth=13), dict(imonth=1.5),
                 dict(iyear=2018), dict(iyear=1997), dict(region=13),
                 dict(imonth=2, iday=30), dict(iday=-1), dict(iday=32)]
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data.validate_events(event_frame(**invalid))

    def test_unknown_casualties_remain_unknown_and_true_zero_is_retained(self):
        d = data.validate_events(event_frame(nkill=-99, nwound=None,
                                            nkillter=0, nwoundte=2))
        self.assertTrue(pd.isna(d.loc[0, 'nkill']))
        self.assertTrue(pd.isna(d.loc[0, 'nwound']))
        self.assertEqual(d.loc[0, 'nkillter'], 0)
        self.assertEqual(d.loc[0, 'nwoundte'], 2)

    def test_coordinate_status_distinguishes_missing_double_zero_and_single_zero(self):
        d = pd.DataFrame({'latitude': [np.nan, 1, 91, 1, 0, 0, 23, -90],
                          'longitude': [2, np.nan, 2, -181, 0, 100, 0, 180]})
        self.assertEqual(data.coordinate_status(d).tolist(),
                         ['缺失', '缺失', '超出范围', '超出范围',
                          '双零待核实', '有效', '有效', '有效'])

    def test_monthly_panel_is_sorted_complete_and_preserves_regional_zero(self):
        d = pd.DataFrame({'month': pd.to_datetime(['2017-03-01', '2017-01-01',
                                                 '2017-02-01', '2017-03-01']),
                          'region': [2, 1, 1, 2]})
        panel = data.monthly_counts(d, '2017-01-01', '2017-03-01')
        pd.testing.assert_index_equal(panel.index,
                                      pd.date_range('2017-01-01', periods=3,
                                                    freq='MS', name='月份'))
        self.assertEqual(panel.columns.tolist(), list(data.REGIONS.values()))
        np.testing.assert_array_equal(panel.sum(axis=1), [1, 1, 2])
        np.testing.assert_array_equal(panel['北美'], [1, 1, 0])
        np.testing.assert_array_equal(panel['中美洲与加勒比'], [0, 0, 2])
        self.assertTrue((panel['大洋洲'] == 0).all())

    def test_global_missing_month_cannot_be_silently_filled_with_zero(self):
        d = pd.DataFrame({'month': pd.to_datetime(['2017-01-01', '2017-03-01']),
                          'region': [1, 2]})
        with self.assertRaisesRegex(ValueError, '整月数据缺失'):
            data.monthly_counts(d, '2017-01-01', '2017-03-01')


class ScoreConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic_scores.csv'
        self.scores = pd.DataFrame({'eventid': [101, 102, 103, 104, 105],
                                    '危害等级': [1, 2, 3, 4, 5],
                                    '危害得分': [.95, .75, .55, .35, .15],
                                    '数据完整率': [1.] * 5})
        self.events = pd.DataFrame({'eventid': [105, 103, 101, 104, 102]})

    def connect(self, scores):
        scores.to_csv(self.path, index=False)
        return data.connect_scores(self.events, self.path)

    def test_join_matches_event_identifier_not_row_position(self):
        result = self.connect(self.scores)
        self.assertEqual(result.eventid.tolist(), self.events.eventid.tolist())
        np.testing.assert_array_equal(result['危害等级'], [5, 3, 1, 4, 2])
        np.testing.assert_allclose(result['危害得分'], [.15, .55, .95, .35, .75])

    def test_duplicate_and_different_event_sets_are_rejected(self):
        duplicate = pd.concat([self.scores, self.scores.iloc[[0]]], ignore_index=True)
        changed = self.scores.copy()
        changed.loc[0, 'eventid'] = 999
        for scores in [duplicate, changed, self.scores.iloc[:-1]]:
            with self.subTest(events=scores.eventid.tolist()):
                with self.assertRaisesRegex(ValueError, '事件集合不一致'):
                    self.connect(scores)

    def test_reversed_level_direction_is_rejected(self):
        scores = self.scores.copy()
        scores['危害得分'] = scores['危害得分'].iloc[::-1].to_numpy()
        with self.assertRaisesRegex(ValueError, '一级最严重'):
            self.connect(scores)

    def test_invalid_score_or_level_is_rejected(self):
        for col, val in [('危害得分', 1.01), ('危害得分', -.01),
                         ('危害得分', np.nan), ('危害等级', 0), ('危害等级', 6)]:
            scores = self.scores.copy()
            scores.loc[0, col] = val
            with self.subTest(column=col, value=val), self.assertRaises(ValueError):
                self.connect(scores)


class ForecastMathTests(unittest.TestCase):
    def test_baselines_predict_twelve_steps_from_one_cutoff(self):
        history = np.arange(1., 61.)
        np.testing.assert_array_equal(forecast.predict(history, '季节朴素'), history[-12:])
        np.testing.assert_allclose(forecast.predict(history, '近12月均值'),
                                   np.repeat(history[-12:].mean(), 12))
        # 超过十二步时，季节朴素重复完整季节而非逐次使用未来真值。
        np.testing.assert_array_equal(forecast.predict(history, '季节朴素', 24),
                                      np.tile(history[-12:], 2))

    def test_constant_history_produces_nonnegative_twelve_step_forecasts(self):
        for value in [0., 8.]:
            for model in forecast.MODELS:
                with self.subTest(value=value, model=model):
                    pred = forecast.predict(np.full(60, value), model)
                    self.assertEqual(pred.shape, (12,))
                    np.testing.assert_array_equal(pred, np.full(12, value))

    def test_invalid_forecast_history_is_rejected(self):
        for values in [np.ones(23), np.r_[np.ones(23), -1],
                       np.r_[np.ones(23), np.nan], np.r_[np.ones(23), np.inf]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                forecast.predict(values, '季节朴素')

    def test_unknown_model_is_rejected_even_for_constant_history(self):
        for values in [np.zeros(60), np.ones(60), np.arange(60.)]:
            with self.subTest(constant=np.ptp(values) == 0):
                with self.assertRaisesRegex(ValueError, '未知模型'):
                    forecast.predict(values, '不存在的模型')

    def test_intervals_are_nonnegative_nested_and_quarter_specific(self):
        errors = np.tile(np.repeat([1., 2., 3., 4.], 3), (5, 1))
        prediction = np.ones(12)
        bounds = forecast.interval(prediction, errors)
        self.assertTrue(np.all(bounds['下限95'] >= 0))
        self.assertTrue(np.all(bounds['下限95'] <= bounds['下限80']))
        self.assertTrue(np.all(bounds['下限80'] <= prediction))
        self.assertTrue(np.all(prediction <= bounds['上限80']))
        self.assertTrue(np.all(bounds['上限80'] <= bounds['上限95']))
        np.testing.assert_allclose(bounds['上限95'] - prediction,
                                   norm.ppf(.975) * np.repeat([1., 2., 3., 4.], 3))

    def test_annual_interval_retains_correlated_monthly_errors(self):
        # 正相关误差在年内累积；反相关误差可相互抵消。
        same_sign = np.ones((5, 12))
        cancelling = np.tile([1., -1.] * 6, (5, 1))
        positive = forecast.interval(100., same_sign, annual=True)
        negative = forecast.interval(100., cancelling, annual=True)
        self.assertAlmostEqual(float(positive['上限95']) - 100., norm.ppf(.975) * 12)
        self.assertEqual(float(negative['上限95']), 100.)
        self.assertEqual(float(negative['下限95']), 100.)
        monthly = forecast.interval(np.full(12, 100. / 12), cancelling)
        self.assertGreater(monthly['上限95'].sum(), float(negative['上限95']))

    def test_intervals_require_complete_finite_years(self):
        for errors in [np.ones(12), np.ones((5, 11)), np.full((5, 12), np.nan)]:
            with self.subTest(shape=errors.shape), self.assertRaises(ValueError):
                forecast.interval(np.ones(12), errors)

    def test_reconciliation_sums_to_global_and_handles_zero_regional_forecasts(self):
        result = forecast.reconcile([[2, 6], [0, 0], [1, 3]], [16, 20, 0], [.25, .75])
        np.testing.assert_allclose(result, [[4, 12], [5, 15], [0, 0]])
        np.testing.assert_allclose(result.sum(axis=1), [16, 20, 0])
        self.assertTrue(np.all(result >= 0))

    def test_metrics_do_not_divide_by_zero_for_empty_event_count(self):
        m = forecast.metrics([0, 0], [1, 3])
        self.assertEqual(m['MAE'], 2.)
        self.assertAlmostEqual(m['RMSE'], np.sqrt(5.))
        self.assertTrue(np.isnan(m['WAPE']))


class ForecastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = seasonal_panel()
        cls.calls = []

        def record(y, model, horizon=12):
            cls.calls.append((y.index.min(), y.index.max(), model, horizon))
            return cheap_predict(y, model, horizon)

        with patch.object(forecast, 'predict', side_effect=record):
            cls.result = forecast.run_forecasts(cls.panel, progress=lambda _: None)

    def test_every_fit_uses_a_year_end_cutoff_and_twelve_steps(self):
        self.assertTrue(self.calls)
        for start, end, model, horizon in self.calls:
            with self.subTest(end=end, model=model):
                self.assertEqual((end.month, end.day), (12, 1))
                self.assertLessEqual(end, pd.Timestamp('2017-12-01'))
                self.assertEqual(horizon, 12)
        split = self.result['时间划分'].set_index('预测年份')
        self.assertEqual(split.loc[2017, '训练结束'], pd.Timestamp('2016-12-01'))
        self.assertEqual(split.loc[2018, '训练结束'], pd.Timestamp('2017-12-01'))
        self.assertTrue((split['预测步数'] == 12).all())

    def test_2018_output_has_all_regions_twelve_months_and_consistent_totals(self):
        monthly = self.result['2018月度预测']
        expected_names = {'全球', *data.REGIONS.values()}
        self.assertEqual(set(monthly['地区']), expected_names)
        self.assertEqual(len(monthly), 13 * 12)
        for name, group in monthly.groupby('地区'):
            self.assertEqual(group['月份'].tolist(),
                             pd.date_range('2018-01-01', periods=12, freq='MS').tolist())
            self.assertEqual(group['步长'].tolist(), list(range(1, 13)))
        table = monthly.pivot(index='月份', columns='地区', values='预测事件数')
        np.testing.assert_allclose(table['全球'], table.drop(columns='全球').sum(axis=1))
        annual = self.result['2018年度预测'].set_index('地区')['预测事件数']
        pd.testing.assert_series_equal(monthly.groupby('地区')['预测事件数'].sum(),
                                       annual.sort_index(), check_names=False)

    def test_holdout_reports_both_baselines_and_interval_coverage(self):
        scores = self.result['2017回测指标']
        self.assertEqual(len(scores), 13 * 3)
        self.assertEqual(set(scores['方案']), {'冻结方案', '季节朴素基线', '近12月均值基线'})
        selected = scores[scores['方案'].eq('冻结方案')]
        self.assertTrue(selected['覆盖率80'].between(0, 1).all())
        self.assertTrue(selected['覆盖率95'].between(0, 1).all())
        np.testing.assert_allclose(selected['MAE'], 0, atol=1e-12)
        calibration = self.result['区间误差尺度']
        self.assertEqual(set(calibration.loc[calibration['用途'].eq('2017测试'), '历史完整年度数']), {5})
        self.assertEqual(set(calibration.loc[calibration['用途'].eq('2018预测'), '历史完整年度数']), {6})

    def test_2017_observations_do_not_change_model_selection_or_holdout_predictions(self):
        changed = self.panel.copy()
        changed.loc['2017-01-01':] *= 100
        with patch.object(forecast, 'predict', side_effect=cheap_predict):
            second = forecast.run_forecasts(changed, progress=lambda _: None)
        pd.testing.assert_frame_equal(self.result['模型选择'], second['模型选择'])
        cols = ['地区', '月份', '预测事件数', '下限80', '上限80', '下限95', '上限95']
        pd.testing.assert_frame_equal(self.result['2017逐月回测'][cols],
                                       second['2017逐月回测'][cols])
        annual_cols = ['地区', '方案', '年度预测事件数',
                       '年度下限80', '年度上限80', '年度下限95', '年度上限95']
        pd.testing.assert_frame_equal(self.result['2017回测指标'][annual_cols],
                                       second['2017回测指标'][annual_cols])
        # 新的留出观测应改变最终预测与最终区间，证明扰动确实进入最终训练段。
        self.assertFalse(np.allclose(self.result['2018月度预测']['预测事件数'],
                                     second['2018月度预测']['预测事件数']))

    def test_zero_2017_regional_count_has_undefined_annual_change(self):
        panel = self.panel[['北美', '东亚']].copy()
        panel.loc['2017-01-01':, '北美'] = 0
        with patch.object(forecast, 'predict', side_effect=cheap_predict):
            with np.errstate(divide='raise', invalid='raise'):
                result = forecast.run_forecasts(panel, progress=lambda _: None)
        annual = result['2018年度预测'].set_index('地区')
        self.assertEqual(annual.loc['北美', '2017实际事件数'], 0)
        self.assertTrue(pd.isna(annual.loc['北美', '相对2017变化率']))
        self.assertTrue(np.isfinite(annual.loc['东亚', '相对2017变化率']))

    def test_annual_holdout_metrics_use_complete_validation_year_errors(self):
        panel = self.panel[['北美', '东亚']].astype(float).copy()
        panel += np.arange(len(panel))[:, None] * .1
        with patch.object(forecast, 'predict', side_effect=cheap_predict):
            result = forecast.run_forecasts(panel, progress=lambda _: None)
        scores = result['2017回测指标'].query("方案 == '冻结方案'").set_index('地区')
        for name, row in scores.iterrows():
            monthly = result['2017逐月回测'].query('地区 == @name')
            actual = monthly['实际事件数'].sum()
            predicted = monthly['预测事件数'].sum()
            self.assertAlmostEqual(row['年度实际事件数'], actual)
            self.assertAlmostEqual(row['年度预测事件数'], predicted)
            self.assertAlmostEqual(row['年度误差_预测减实际'], predicted - actual)
            validation = result['选模期协调预测'].query('地区 == @name').sort_values('月份')
            self.assertEqual(set(validation['月份'].dt.year), set(range(2012, 2017)))
            errors = validation['误差_实际减预测'].to_numpy().reshape(5, 12)
            expected = forecast.interval(predicted, errors, annual=True)
            for cov in [80, 95]:
                lo, hi = float(expected[f'下限{cov}']), float(expected[f'上限{cov}'])
                self.assertAlmostEqual(row[f'年度下限{cov}'], lo)
                self.assertAlmostEqual(row[f'年度上限{cov}'], hi)
                self.assertIsInstance(row[f'年度覆盖{cov}'], (bool, np.bool_))
                self.assertEqual(row[f'年度覆盖{cov}'], lo <= actual <= hi)

    def test_failed_candidate_is_logged_and_excluded(self):
        def broken(y, model, horizon=12):
            if model == '低阶SARIMA':
                raise ValueError('合成的不收敛错误')
            return cheap_predict(y, model, horizon)

        with patch.object(forecast, 'predict', side_effect=broken):
            result = forecast.run_forecasts(self.panel[['北美']], progress=lambda _: None)
        rows = result['模型选择'].query("模型 == '低阶SARIMA'")
        self.assertTrue((~rows['有效']).all())
        self.assertTrue((~rows['入选']).all())
        self.assertEqual(set(result['拟合异常']['处理']), {'候选排除'})
        self.assertEqual(len(result['拟合异常']), 2 * 5)

    def test_holdout_fit_failure_falls_back_without_reselecting(self):
        def selected_ets_fails(y, model, horizon=12):
            if model == '阻尼季节ETS':
                if y.index.max() == pd.Timestamp('2016-12-01'):
                    raise ValueError('合成的最终拟合错误')
                return np.asarray(y)[-12:]
            return cheap_predict(y, model, horizon) + 10

        with patch.object(forecast, 'predict', side_effect=selected_ets_fails):
            result = forecast.run_forecasts(self.panel[['北美']], progress=lambda _: None)
        winners = result['模型选择'].query('入选')
        self.assertEqual(set(winners['模型']), {'阻尼季节ETS'})
        self.assertEqual(set(result['拟合异常']['处理']), {'明确回退季节朴素'})
        self.assertEqual(set(result['拟合异常']['预测年份']), {2017})


if __name__ == '__main__':
    unittest.main(verbosity=2)
