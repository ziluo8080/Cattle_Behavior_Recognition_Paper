"""Count the preserved official loader stages; no model import or training.

Requires the historical numpy/pandas environment and the 40 original CSVs.
This counter is provided for raw-data holders; it was not executed for v10.
"""
import argparse,hashlib,json,pathlib

def main():
    import numpy as np
    import pandas as pd
    ap=argparse.ArgumentParser()
    ap.add_argument('--sensor-data-dir',required=True)
    ap.add_argument('--output',default='OFFICIAL_TARGET_FLOW_RECOUNT.json')
    args=ap.parse_args();root=pathlib.Path(args.sensor_data_dir);rows=[];members=[];keys=[]
    for cow in range(1,11):
        tag=f'T{cow:02d}';cid=f'C{cow:02d}'
        paths={'uwb':root/'main_data/uwb'/tag/f'{tag}_0725.csv',
               'hd':root/'sub_data/head_direction'/tag/f'{tag}_0725.csv',
               'akl':root/'main_data/ankle'/cid/f'{cid}_0725.csv',
               'label':root/'behavior_labels/individual'/f'{cid}_0725.csv'}
        for name,p in paths.items():
            members.append({'cow_id':cow,'stream':name,'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        u,h,a,l=(pd.read_csv(paths[k]) for k in ['uwb','hd','akl','label'])
        rec={'cow_id':cow,'raw_uwb_rows':len(u),'raw_uwb_unique_timestamp':int(u.timestamp.nunique()),'raw_label_rows':len(l),'raw_hd_rows':len(h),'raw_akl_rows':len(a)}
        # Exactly the inherited target loader; NOT the new past-only windows.
        h['timestamp']=np.floor(h['timestamp']).astype(int)
        h=h.groupby('timestamp').mean().reset_index()
        rec['hd_second_rows']=len(h)
        h=h[h['timestamp'].isin(u['timestamp'].values)].astype('float64')
        l=l[l['timestamp'].isin(u['timestamp'].values)]
        rec['label_at_uwb_rows']=len(l);rec['hd_at_uwb_rows']=len(h)
        a=a.drop(columns=['datetime']).astype('float64')
        h=pd.merge_asof(h,a,on='timestamp',direction='nearest')
        l=l[l['behavior']!=0].copy();known=l.values[:,0]
        rec['known_label_rows']=len(l)
        u=u[u['timestamp'].isin(known)];h=h[h['timestamp'].isin(known)]
        rec['uwb_after_known_label_rows']=len(u);rec['hd_akl_after_known_label_rows']=len(h)
        merged=pd.merge(pd.merge(u,h,on='timestamp'),l,on='timestamp').drop(columns=['datetime'])
        rec['merged_before_dropna_rows']=len(merged)
        final=merged.dropna().drop(columns=['accel_norm','relative_angle'])
        rec['merged_dropna_removed_rows']=len(merged)-len(final)
        rec['final_rows']=len(final);rec['final_unique_timestamp']=int(final.timestamp.nunique())
        rows.append(rec);keys.extend((cow,int(v)) for v in final.timestamp.values)
    result={'source':'preserved official uwb_hd_aa_pre_loader operations; independent counter','model_execution':False,'per_cow':rows,'raw_uwb_rows':sum(r['raw_uwb_rows'] for r in rows),'final_rows':sum(r['final_rows'] for r in rows),'final_unique_cow_timestamp_keys':len(set(keys)),'members':members,'note':'Stage counts are successive operations; overlapping missingness causes must not be added as independent exclusions. The official target inheritance uses nearest ankle matching; v10 new history windows are separately constrained to past sources.'}
    pathlib.Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':args.output,'raw_uwb_rows':result['raw_uwb_rows'],'final_unique_keys':len(set(keys))}))

if __name__=='__main__':main()
