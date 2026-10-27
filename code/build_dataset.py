"""Step 1: derive the 289-compound analysis set, descriptors, and selection-bias comparison."""
import json, numpy as np, pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, inchi
RDLogger.DisableLog('rdApp.*')
cfg = json.load(open('config.json'))
df = pd.read_csv(cfg['dataset_file'])
flow = {'source_compounds': len(df)}
rat = df['rat_oral_LD50']; flow['rat_oral_LD50'] = int(rat.notna().sum())
hum = df[['man_oral_TDLo','women_oral_TDLo','human_oral_TDLo']]
flow['rat_oral_LD50_with_any_human_oral_TDLo'] = int((rat.notna() & hum.notna().any(axis=1)).sum())
ratdf = df[rat.notna()].copy()
hum_r = ratdf[['man_oral_TDLo','women_oral_TDLo','human_oral_TDLo']]
ratdf['has_human'] = hum_r.notna().any(axis=1)

def feats(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return dict(parsed=False)
    frags = Chem.GetMolFrags(m, asMols=True)
    big = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    syms = {a.GetSymbol() for a in m.GetAtoms()}
    metals = set("Li Na K Rb Cs Be Mg Ca Sr Ba Al Ga In Tl Sn Pb Bi Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg La Ce Nd Sm Gd U Th As Sb Se Te B Si Ge".split())
    ik = ''
    try: ik = inchi.MolToInchiKey(big).split('-')[0]
    except Exception: pass
    return dict(parsed=True, MW=Descriptors.MolWt(big), heavy=big.GetNumHeavyAtoms(), logP=Crippen.MolLogP(big),
                TPSA=rdMolDescriptors.CalcTPSA(big), arom_rings=rdMolDescriptors.CalcNumAromaticRings(big),
                n_frag=len(frags), has_C='C' in syms, metal_metalloid=bool(syms & metals), has_P='P' in syms,
                has_halogen=bool(syms & {'F','Cl','Br','I'}), has_N='N' in syms, has_S='S' in syms,
                charged=any(a.GetFormalCharge()!=0 for a in m.GetAtoms()),
                acid=big.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')),
                inchikey_conn=ik)
F = pd.DataFrame([feats(s) for s in ratdf['SMILES']], index=ratdf.index)
ratdf = pd.concat([ratdf, F], axis=1)
ratdf['n_endpoints_measured'] = ratdf.drop(columns=['RTECS_ID','SMILES'] + list(F.columns) + ['has_human']).notna().sum(axis=1)
flow['rat_oral_unparsed_smiles'] = int((~ratdf['parsed']).sum())

A = ratdf[ratdf.has_human].copy()
h = A[['man_oral_TDLo','women_oral_TDLo','human_oral_TDLo']]
A['x'] = A['rat_oral_LD50']; A['y'] = h.mean(axis=1); A['y_first'] = h.bfill(axis=1).iloc[:,0]
A['n_human_labels'] = h.notna().sum(axis=1)
A['label_pattern'] = h.notna().apply(lambda r: '+'.join([n.split('_')[0] for n,v in r.items() if v]), axis=1)
A['human_label_sd'] = h.std(axis=1, ddof=1)
A['human_label_range'] = h.max(axis=1) - h.min(axis=1)
flow['analysis_n'] = len(A)
flow['duplicate_RTECS_ID'] = int(A.RTECS_ID.duplicated().sum())
flow['duplicate_SMILES'] = int(A.SMILES.duplicated().sum())
flow['duplicate_InChIKey_connectivity_within_289'] = int(A.inchikey_conn[A.inchikey_conn!=''].duplicated().sum())
allk = ratdf.inchikey_conn[ratdf.inchikey_conn!='']
flow['duplicate_InChIKey_connectivity_within_10190'] = int(allk.duplicated().sum())
flow['n_multi_label'] = int((A.n_human_labels>1).sum())
flow['label_patterns'] = A.label_pattern.value_counts().to_dict()
flow['multi_label_sd_median'] = float(A.human_label_sd.dropna().median())
flow['multi_label_range_median'] = float(A.human_label_range[A.n_human_labels>1].median())
flow['multi_label_range_max'] = float(A.human_label_range[A.n_human_labels>1].max())
A[['RTECS_ID','SMILES','x','y','y_first','man_oral_TDLo','women_oral_TDLo','human_oral_TDLo','n_human_labels','label_pattern',
   'MW','heavy','logP','TPSA','arom_rings','n_frag','has_C','metal_metalloid','has_P','has_halogen','has_N','has_S','charged','acid','n_endpoints_measured']].to_csv('../out/analysis_set_289.csv', index=False)

# selection comparison: 289 vs other rat-oral-LD50 compounds
O = ratdf[~ratdf.has_human]
rows = []
def cont(name, col):
    a, b = A[col].dropna(), O[col].dropna()
    smd = (a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
    rows.append(dict(variable=name, kind='continuous', analysis_n=len(a), other_n=len(b),
        analysis=f"{a.median():.2f} [{a.quantile(.25):.2f}, {a.quantile(.75):.2f}]",
        other=f"{b.median():.2f} [{b.quantile(.25):.2f}, {b.quantile(.75):.2f}]",
        smd=smd, p=stats.mannwhitneyu(a,b).pvalue, ks_p=stats.ks_2samp(a,b).pvalue))
def binv(name, col):
    a = A[col][A.parsed].astype(bool); b = O[col][O.parsed].astype(bool)
    pa, pb = a.mean(), b.mean()
    smd = (pa-pb)/np.sqrt((pa*(1-pa)+pb*(1-pb))/2)
    tab = [[a.sum(), (~a).sum()],[b.sum(), (~b).sum()]]
    rows.append(dict(variable=name, kind='binary', analysis_n=len(a), other_n=len(b),
        analysis=f"{100*pa:.1f}%", other=f"{100*pb:.1f}%", smd=smd, p=stats.fisher_exact(tab)[1], ks_p=np.nan))
cont('Rat oral LD50 (-log10 mol/kg)', 'rat_oral_LD50')
for n,c in [('Molecular weight (largest fragment)','MW'),('Heavy atoms','heavy'),('Crippen logP','logP'),('TPSA','TPSA'),('Aromatic rings','arom_rings'),('Endpoints measured per compound','n_endpoints_measured')]:
    cont(n,c)
for n,c in [('Multi-fragment (salt/mixture)','n_frag'),('Contains carbon','has_C'),('Metal/metalloid-containing','metal_metalloid'),('Phosphorus-containing','has_P'),('Halogen-containing','has_halogen'),('Nitrogen-containing','has_N'),('Sulfur-containing','has_S'),('Carboxylic acid','acid'),('Formally charged atom','charged')]:
    if c=='n_frag':
        A['_t']=A.n_frag>1; O=O.assign(_t=O.n_frag>1); c='_t'
    binv(n,c)
pd.DataFrame(rows).to_csv('../out/selection_comparison.csv', index=False)

# analysis-set summary
S = {}
for c in ['x','y','y_first']:
    S[c] = dict(mean=A[c].mean(), sd=A[c].std(), median=A[c].median(), q25=A[c].quantile(.25), q75=A[c].quantile(.75), min=A[c].min(), max=A[c].max())
S['corr_pearson'] = float(np.corrcoef(A.x, A.y)[0,1]); S['spearman'] = float(stats.spearmanr(A.x, A.y)[0])
S['other_x'] = dict(mean=O.rat_oral_LD50.mean(), sd=O.rat_oral_LD50.std(), median=O.rat_oral_LD50.median(), n=len(O))
json.dump(dict(flow=flow, summary=S), open('../out/dataset_summary.json','w'), indent=1, default=float)
print(json.dumps(flow, indent=1, default=float)); print(pd.DataFrame(rows).round(3).to_string())
print(json.dumps(S, indent=1, default=float))
