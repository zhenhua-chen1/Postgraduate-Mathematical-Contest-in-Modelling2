#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2018 年 C 题第二问：案件关联、线索分组与表 2 嫌疑排序。

Spyder 中打开本文件后点击绿色箭头即可，全部结果写入同目录“优化结果”。
线索组是统计模型假设，不能直接解释为已查明的真实组织。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

# 相邻模块和附件始终从脚本目录读取，不依赖 Spyder 顶部的工作目录。
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from association import build_graph, cluster_graph, clustering_metrics, score_candidates, train_and_validate
from reporting import export_results

SEED = 2025
QUERY_IDS = [201701090031, 201702210037, 201703120023, 201705050009,
             201705050010, 201707010028, 201707020006, 201708110018,
             201711010006, 201712010003]
USE_COLUMNS = [
    'eventid', 'iyear', 'imonth', 'iday', 'country', 'country_txt', 'region',
    'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity',
    'attacktype1', 'targtype1', 'targsubtype1', 'weaptype1', 'weapsubtype1',
    'success', 'suicide', 'ishostkid', 'multiple', 'extended', 'gname', 'gname2',
    'gname3', 'guncertain1', 'individual', 'claimed', 'claim2', 'claim3',
    'nkill', 'nkillter', 'nwound', 'nwoundte', 'nhostkid', 'property',
    'propextent', 'propvalue',
]
# 这类标签是泛称，不能把同一称呼下的多起事件都当成“同一个组织”。
GENERIC_NAMES = {'maoists', 'anarchists', 'separatists', 'gunmen', 'militants',
                 'rebels', 'anti-government extremists', 'palestinian extremists',
                 'muslim extremists', 'islamist extremists', 'islamic extremists',
                 'unidentified', 'unknown', 'unaffiliated individual(s)',
                 'mapuche activists', 'left-wing extremists', 'right-wing extremists',
                 'anti-immigrant extremists', 'rohingya extremists', 'jihadi-inspired extremists',
                 'shuar extremists', 'neo-nazi extremists', 'fulani extremists',
                 'cossack separatists', 'anti-police extremists', 'tribesmen',
                 'ukrainian nationalists', 'white extremists', 'israeli settlers',
                 'israeli extremists', 'supporters of abd rabbuh mansur hadi',
                 'supporters of ali abdullah saleh', 'pro-russia militia', 'kuki tribal militants'}


def progress(message):
    """立即刷新控制台，长步骤开始前也会先给提示。"""
    print(message, flush=True)


def known_name(series):
    return series.notna() & ~series.fillna('').str.strip().str.lower().isin(['', 'unknown'])


def load_data(path):
    """原始附件1为权威输入；历史附件2/3只用于口径与题目编号核对。"""
    full = pd.read_excel(path, usecols=USE_COLUMNS)
    if full.eventid.isna().any() or full.eventid.duplicated().any():
        raise ValueError('附件1的事件编号缺失或重复，请核对原始数据。')
    full['eventid'] = full.eventid.astype('int64')
    period = full.iyear.isin([2015, 2016])
    # 缺失/-9代表“不知道是否认领”，不能填成0。组织名Unknown也不等于未认领。
    any_claim = full[['claimed', 'claim2', 'claim3']].eq(1).any(axis=1)
    unclaimed = full.claimed.eq(0) & ~any_claim
    unknown = ~known_name(full.gname)
    target = full.loc[period & unclaimed].sort_values('eventid').reset_index(drop=True)
    reliable = (known_name(full.gname) & full.guncertain1.eq(0)
                & ~known_name(full.gname2) & ~known_name(full.gname3)
                & full.individual.ne(1) & full.claimed.eq(1))
    generic = full.gname.fillna('').str.strip().str.lower().isin(GENERIC_NAMES)
    known = full.loc[period & reliable & ~generic].sort_values('eventid').reset_index(drop=True)
    assert not set(target.eventid) & set(known.eventid)
    queries = full.set_index('eventid').reindex(QUERY_IDS)
    if queries.iyear.isna().any():
        raise ValueError('附件1缺少表2指定事件。')
    queries = queries.reset_index()
    audit = [('附件1全部事件', len(full)), ('2015—2016全部事件', int(period.sum())),
             ('明确无人认领（主分析）', len(target)),
             ('主分析中组织名Unknown', int((period & unclaimed & unknown).sum())),
             ('主分析中已有归属记录', int((period & unclaimed & ~unknown).sum())),
             ('2015—2016组织名Unknown（旧口径）', int((period & unknown).sum())),
             ('认领状态未知且无人明确认领', int((period & ~any_claim & ~full.claimed.isin([0, 1])).sum())),
             ('可靠已认领记录（剔除泛称前）', int((period & reliable).sum())),
             ('剔除泛称标签记录', int((period & reliable & generic).sum())),
             ('可靠已认领记录（验证来源）', len(known)),
             ('目标与训练来源事件重叠', 0), ('表2事件', len(queries))]
    legacy_diff = pd.DataFrame(columns=['eventid', '差异说明'])
    if (PROJECT_DIR / '附件2.xlsx').exists():
        legacy = pd.read_excel(PROJECT_DIR / '附件2.xlsx', usecols=['eventid', 'claimed'])
        ids, expected = set(legacy.eventid), set(full.loc[period & unknown, 'eventid'])
        legacy_diff = pd.DataFrame([{'eventid': i, '差异说明': '旧附件2未包含的Unknown案件'} for i in sorted(expected-ids)]
                                  + [{'eventid': i, '差异说明': '旧附件2含有但不属于Unknown口径'} for i in sorted(ids-expected)])
        audit += [('旧附件2条数', len(legacy)), ('旧附件2中已宣称负责', int(legacy.claimed.eq(1).sum())),
                  ('旧附件2遗漏Unknown条数', len(expected-ids))]
    if (PROJECT_DIR / '附件3.xlsx').exists():
        ids = pd.read_excel(PROJECT_DIR / '附件3.xlsx', usecols=['eventid']).eventid.astype('int64')
        if len(ids) != 10 or set(ids) != set(QUERY_IDS):
            raise ValueError('旧附件3与真题表2编号不一致。')
    return full, target, known, queries, pd.DataFrame(audit, columns=['口径','事件数']), legacy_diff


def hazard_scores(full, target):
    """复用第一问的回顾性危害标尺；危害不进入案件关联特征。"""
    first = PROJECT_DIR.parent / '2018年C题第1问'
    source = first / '优化结果' / '案件危害分级.csv'
    if source.exists():
        scores = pd.read_csv(source, usecols=['eventid','危害得分'])
        origin = '第一问已保存的连续危害得分（1998—2017全样本回顾性标尺）'
    else:
        path = first / 'main.py'
        if not path.exists():
            raise FileNotFoundError('需要相邻第一问的main.py或案件危害分级.csv。')
        spec = importlib.util.spec_from_file_location('question_one_hazard', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        indicators, groups, _ = module.prepare_indicators(full)
        score, _, _, _ = module.fit_hierarchical_weights(indicators, groups)
        scores = pd.DataFrame({'eventid': full.eventid, '危害得分': score})
        origin = '按第一问相同函数重建的全样本回顾性标尺'
    if scores.eventid.duplicated().any():
        raise ValueError('第一问危害得分有重复编号。')
    joined = target[['eventid']].merge(scores, on='eventid', how='left', validate='one_to_one')
    if joined['危害得分'].isna().any() or not joined['危害得分'].between(0, 1).all():
        raise ValueError('第一问危害得分缺失或超范围，请先更新第一问。')
    return joined['危害得分'].to_numpy(), origin, source


def partition_metrics(a, b):
    """用成对指标描述同组关系，而不让海量不同组真阴性抬高准确率。"""
    # 单件线索可能有上万组，使用仅保存非零交集的频数表，避免巨大的稠密矩阵。
    pairs = pd.DataFrame({'a': np.asarray(a), 'b': np.asarray(b)})
    intersections = pairs.groupby(['a', 'b']).size().to_numpy(dtype=np.int64)
    choose2 = lambda v: (v * (v-1) // 2).sum()
    tp = choose2(intersections)
    pa, pb = choose2(pairs.a.value_counts().to_numpy()), choose2(pairs.b.value_counts().to_numpy())
    return {'ARI': adjusted_rand_score(a,b), 'NMI': normalized_mutual_info_score(a,b),
            '同组对保留率': tp/pa if pa else np.nan, '新划分同组对一致率': tp/pb if pb else np.nan}


def summarize_groups(target, labels, hazard, edges, threshold):
    """累计危害降序编号，并列时用最小事件编号；单件组不进入重复作案前五名。"""
    members = target[['eventid','iyear','imonth','iday','country_txt','provstate','city',
                      'attacktype1','targtype1','weaptype1','claimed','gname']].copy()
    members['cluster'], members['危害得分'] = labels, hazard
    members['受害者死亡'] = (target.nkill - target.nkillter.fillna(0)).clip(lower=0)
    members['受害者受伤'] = (target.nwound - target.nwoundte.fillna(0)).clip(lower=0)
    rows = []
    for c, part in members.groupby('cluster', sort=True):
        rows.append({'cluster': int(c), '事件数': len(part), '累计危害': part['危害得分'].sum(),
                     '平均危害': part['危害得分'].mean(), '最高单次危害': part['危害得分'].max(),
                     '主要国家': part.country_txt.mode().iloc[0], '国家数': part.country_txt.nunique(),
                     '起始事件编号': int(part.eventid.min()), '结束事件编号': int(part.eventid.max()),
                     '已有归属记录数': int(known_name(part.gname).sum()),
                     '已记录受害者死亡': part['受害者死亡'].sum(min_count=1),
                     '已记录受害者受伤': part['受害者受伤'].sum(min_count=1)})
    groups = pd.DataFrame(rows).sort_values(['累计危害','起始事件编号'], ascending=[False,True]).reset_index(drop=True)
    codes = {int(c): f'G{i+1:04d}' for i,c in enumerate(groups.cluster)}
    groups.insert(0,'线索组',groups.cluster.map(codes))
    top = groups.loc[groups['事件数'].ge(2)].head(5).copy()
    if len(top) < 5:
        raise ValueError('阈值下不足5个重复作案候选组，请检查验证与数据。')
    top.insert(0,'嫌疑人代号',np.arange(1,6))
    mapping = dict(zip(top.cluster,top['嫌疑人代号']))
    groups['嫌疑人代号'] = groups.cluster.map(mapping).astype('Int64')
    members['线索组'] = members.cluster.map(codes)
    members['嫌疑人代号'] = members.cluster.map(mapping).astype('Int64')
    members['分组状态'] = np.where(members.cluster.map(members.cluster.value_counts()).eq(1),'单件线索','重复作案候选组')
    degree = np.zeros(len(target), dtype=int)
    inside = edges.loc[edges.score.ge(threshold)]
    if len(inside):
        within = labels[inside.i.to_numpy()] == labels[inside.j.to_numpy()]
        np.add.at(degree,inside.loc[within,'i'],1)
        np.add.at(degree,inside.loc[within,'j'],1)
    members['组内高分关联数'] = degree
    return groups, top, members


def rank_queries(model, queries, target, labels, top, threshold, pair_threshold):
    """各列是固定嫌疑人，单元格填名次；最高3个参照案件平均分为关联支持度。"""
    # 没有符合地理条件的参考案件时，相似度不可计算，不能伪装成0分。
    scores, evidence = np.full((len(queries),5),np.nan), []
    counts = np.zeros((len(queries),5), dtype=int)
    for _, row in top.iterrows():
        suspect, cluster = int(row['嫌疑人代号']), int(row.cluster)
        ref = target.loc[labels == cluster].reset_index(drop=True)
        pairs = score_candidates(model,queries,ref,top_k=3)
        for qi, part in pairs.groupby('query_index'):
            qi = int(qi)
            scores[qi,suspect-1] = part.score.mean()
            counts[qi,suspect-1] = int(part.score.ge(pair_threshold).sum())
            for item in part.itertuples(index=False):
                ri = int(item.reference_index)
                evidence.append({'eventid': int(queries.iloc[qi].eventid), '嫌疑人代号': suspect,
                                 '参照事件编号': int(ref.iloc[ri].eventid), '案件对相似分': float(item.score),
                                 '参照国家': ref.iloc[ri].country_txt})
    ranks = pd.DataFrame({'eventid': queries.eventid})
    support = pd.DataFrame({'eventid': queries.eventid,'事件国家': queries.country_txt})
    for s in range(5):
        ranks[f'{s+1}号嫌疑人'] = pd.Series(pd.NA,index=ranks.index,dtype='Int64')
        support[f'{s+1}号支持度'] = scores[:,s]
        support[f'{s+1}号高分参照数'] = counts[:,s]
    for qi in range(len(queries)):
        accepted = sorted([s for s in range(5) if scores[qi,s] >= threshold and counts[qi,s] >= 2],
                          key=lambda s:(-scores[qi,s],s))
        for rank,s in enumerate(accepted,1):
            ranks.loc[qi,f'{s+1}号嫌疑人'] = rank
    support['判定阈值'] = threshold
    support['达到阈值的候选数'] = ((scores >= threshold) & (counts >= 2)).sum(axis=1)
    support['解释'] = np.where(support['达到阈值的候选数'].gt(0),'按支持度降序填写名次','前五组支持不足，表2留空')
    support.loc[~np.isfinite(scores).any(axis=1),'解释'] = '前五组没有符合地理条件的参照，表2留空'
    sensitivity = [{'阈值': t, '有候选事件数': int(((scores>=t)&(counts>=2)).any(axis=1).sum()),
                    '填写格数': int(((scores>=t)&(counts>=2)).sum()), '表2事件数': len(queries)}
                   for t in sorted(set(np.clip([threshold-.05,threshold,threshold+.05],.01,.999)))]
    return ranks,support,pd.DataFrame(evidence),pd.DataFrame(sensitivity)


def attribution_diagnostic(target, labels):
    """事后核对目标集已有归属记录；这些标签不进入建图，也不用于调参。

    未认领不等于无归属。数据库归属记录仍可能有误，因此本表是代理诊断，
    不能把“与数据库一致”解释成组织身份已经得到证明。
    """
    usable = (known_name(target.gname) & target.guncertain1.eq(0)
              & ~known_name(target.gname2) & ~known_name(target.gname3)
              & target.individual.ne(1)
              & ~target.gname.fillna('').str.strip().str.lower().isin(GENERIC_NAMES))
    part = target.loc[usable]
    if part.empty:
        return pd.DataFrame(columns=['事件数', '组织标签数', '主分析覆盖率', '诊断说明'])
    return pd.DataFrame([{'事件数': len(part), '组织标签数': part.gname.nunique(),
                          '主分析覆盖率': len(part)/len(target),
                          **clustering_metrics(part.gname.to_numpy(), labels[usable.to_numpy()]),
                          '诊断说明': '未认领案件的已有归属代理标签，仅事后核对，不参与训练或调参'}])


def stability_analysis(target,labels,edges,threshold,runs):
    """阈值扰动和80%无放回事件子抽样，不冒称重训Bootstrap。"""
    rng, sensitivity, stability = np.random.default_rng(SEED), [], []
    for t in sorted(set(np.clip([threshold-.05,threshold,threshold+.05],.01,.999))):
        alt = cluster_graph(len(target),edges,float(t))
        sizes = pd.Series(alt).value_counts()
        sensitivity.append({'阈值': t, '线索组数': len(sizes), '单件事件比例': (sizes==1).sum()/len(target),
                            '最大组事件数': sizes.max(), **partition_metrics(labels,alt)})
    for run in range(runs):
        keep = np.sort(rng.choice(len(target),size=int(.8*len(target)),replace=False))
        mapping = np.full(len(target),-1,dtype=int)
        mapping[keep] = np.arange(len(keep))
        sub = edges.loc[(mapping[edges.i.to_numpy()]>=0)&(mapping[edges.j.to_numpy()]>=0)].copy()
        sub['i'],sub['j'] = mapping[sub.i.to_numpy()],mapping[sub.j.to_numpy()]
        alt = cluster_graph(len(keep),sub,threshold)
        sizes = pd.Series(alt).value_counts()
        stability.append({'重复次数':run+1,'保留事件数':len(keep),'线索组数':len(sizes),
                          '最大组事件数':sizes.max(),**partition_metrics(labels[keep],alt)})
        if run==0 or (run+1)%5==0 or run+1==runs:
            progress(f'      事件子抽样稳定性 {run+1}/{runs}')
    return pd.DataFrame(sensitivity),pd.DataFrame(stability)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('附件1.xlsx'))
    parser.add_argument('--output',type=Path,default=Path('优化结果'))
    parser.add_argument('--stability-runs',type=int,default=20,help='80%事件子抽样次数，默认20')
    parser.add_argument('--neighbors',type=int,default=20,help='每起案件候选近邻数，默认20')
    args = parser.parse_args(argv)
    if args.stability_runs<1 or args.neighbors<2:
        parser.error('stability-runs至少1，neighbors至少2。')
    path = args.input if args.input.is_absolute() else PROJECT_DIR/args.input
    output = args.output if args.output.is_absolute() else PROJECT_DIR/args.output
    if output.resolve()==PROJECT_DIR:
        parser.error('请使用独立结果文件夹，例如优化结果。')
    started = perf_counter()
    progress('2018年C题第二问开始运行：模型给出待核查线索组。')
    progress('[1/8] 读取原始数据并核对筛选口径……')
    full,target,known,queries,audit,legacy_diff = load_data(path)
    progress(f'      待关联 {len(target):,} 起，可靠已认领来源 {len(known):,} 起，表2共10起。')
    progress('[2/8] 读取第一问连续危害得分……')
    hazard,hazard_origin,hazard_source = hazard_scores(full,target)
    progress('[3/8] 学习案件对相似度，进行组织与时间留出验证……')
    model,validation,config = train_and_validate(known,seed=SEED,progress=progress)
    threshold = float(config['threshold'])
    progress(f'      验证集关联阈值 {threshold:.4f}。')
    for row in validation['pair_validation'].itertuples(index=False):
        if row.model == 'full_pair_model' and row.split.startswith('test_'):
            label = '独立组织测试' if row.split == 'test_unseen_organizations' else '2016年时间测试'
            progress(f'      {label}：精确率 {row.precision:.1%}，召回率 {row.recall:.1%}（抽样案件对）。')
    progress('[4/8] 生成候选案件对并对未认领事件分组……')
    edges = build_graph(model,target,threshold=max(.01,threshold-.05),neighbors=args.neighbors,seed=SEED)
    labels = cluster_graph(len(target),edges,threshold)
    groups,top,members = summarize_groups(target,labels,hazard,edges,threshold)
    progress(f'      形成 {len(groups):,} 个线索组，最大组 {groups["事件数"].max():,} 起。')
    progress('[5/8] 计算表2支持度、拒识与名次……')
    group_threshold = float(config.get('group_threshold',threshold))
    ranks,support,evidence,rank_sensitivity = rank_queries(model,queries,target,labels,top,group_threshold,threshold)
    progress('[6/8] 检查阈值与事件抽样敏感性……')
    sensitivity,stability = stability_analysis(target,labels,edges,threshold,args.stability_runs)
    severity = groups.loc[groups['事件数'].ge(2),['线索组','事件数','累计危害','平均危害','最高单次危害']].copy()
    for col in ['累计危害','平均危害','最高单次危害']:
        severity[col+'名次'] = severity[col].rank(method='min',ascending=False).astype(int)
    tables = {'表2嫌疑排序':ranks,'嫌疑支持度':support,'前五线索组':top.drop(columns='cluster'),
              '全部线索组':groups.drop(columns='cluster'),'案件分组明细':members.drop(columns='cluster'),
              '相似案件证据':evidence,'筛选口径核对':audit,'旧附件差异':legacy_diff,
              **validation,'目标已有归属核对':attribution_diagnostic(target,labels),
              '阈值敏感性':sensitivity,'事件子抽样稳定性':stability,
              '拒识阈值敏感性':rank_sensitivity,'危害排序口径':severity}
    sizes = pd.Series(labels).value_counts()
    summary = {'random_seed':SEED,'target_events':len(target),
               'python_version':sys.version.split()[0],
               'package_versions':{name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','networkx','matplotlib','openpyxl','threadpoolctl']},
               'target_definition':'2015/2016 claimed=0且claim2/claim3均非1（包括有归属但未认领的案件）',
               'known_source_events':len(known),'query_events':len(queries),'threshold':threshold,
               'group_threshold':group_threshold,'neighbors':args.neighbors,'clusters':len(groups),
               'singletons':int((sizes==1).sum()),'largest_cluster':int(sizes.max()),
               'stability_runs':args.stability_runs,'median_subsample_ARI':float(stability.ARI.median()),
               'query_events_with_candidates':int(support['达到阈值的候选数'].gt(0).sum()),
               'hazard_source':hazard_origin,'config':config,'input_sha256':sha256(path),
               'hazard_csv_sha256':sha256(hazard_source) if hazard_source.exists() else None,
               'limitations':['已认领与未认领案件可能有分布差异，已知组织标签仍可能含错误。',
                              '相似分及组支持度不是某组织真实作案概率；低支持度留空。',
                              '候选近邻与社区分辨率影响分组，一个线索组不必对应唯一组织。',
                              '第一问得分是全时期回顾性标尺，不是严格的当时可用危害预测。',
                              '子抽样检验固定模型和候选图的划分变化，不是重训Bootstrap。']}
    progress('[7/8] 校验并生成 Excel、CSV、图表……')
    assert len(members)==len(target) and members.eventid.is_unique
    assert int(groups['事件数'].sum())==len(target)
    assert np.isclose(groups['累计危害'].sum(),hazard.sum())
    assert list(ranks.eventid)==QUERY_IDS
    for row in ranks.iloc[:,1:].itertuples(index=False,name=None):
        filled = [int(x) for x in row if not pd.isna(x)]
        assert sorted(filled)==list(range(1,len(filled)+1))
    export_results(tables,summary,output)
    progress('[8/8] 全部完成。')
    progress(ranks.astype('string').fillna('—').to_string(index=False))
    progress(f'耗时 {perf_counter()-started:.1f} 秒；全部结果已保存到：{output.resolve()}')
    return summary


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
