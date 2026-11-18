"""Registered architectural interventions; no dormant trainable parameters."""
import torch
from torch import nn
from scripts.sensor_convtran_endpoint_interaction_20261022 import EndpointInteractionConvTran
from scripts.sensor_convtran_context_model_20261022 import MaskedConvTran,masked_mean

ARMS=('full','no_eri','short_10_10','long_60_60','dynamic_only','state_only','no_uwb','no_ankle')
class AblationModel(EndpointInteractionConvTran):
 def __init__(self,cfg,prior,arm):
  assert arm in ARMS;super().__init__(cfg,prior);self.arm=arm
  self.dynamic_steps=600 if arm=='long_60_60' else 100
  self.state_steps=10 if arm=='short_10_10' else 60
  opts=dict(width=cfg['width'],heads=cfg['heads'],ff=cfg['ff'],dropout=cfg['dropout'])
  if self.dynamic_steps!=100:self.dynamic=MaskedConvTran(8,self.dynamic_steps,**opts)
  if self.state_steps!=60:self.state=MaskedConvTran(26,self.state_steps,**opts)
  if arm in ('no_eri','dynamic_only','state_only'):
   del self.dynamic_endpoint_from_state,self.state_endpoint_from_dynamic
  width=cfg['width']
  if arm=='dynamic_only':
   del self.state,self.context
  if arm=='state_only':
   del self.dynamic
 def forward(self,batch):
  arm=self.arm
  if arm!='state_only':
   dv=batch['dynamic'][:,:,6]>.5
   phys=torch.where(dv[:,:,None],batch['dynamic'][:,:,:6],0.)
   age=torch.where(dv[:,:,None],batch['dynamic'][:,:,7:].clamp(0,.2),.2)
   d=self.dynamic(torch.cat((phys,dv[:,:,None].float(),age),-1),dv)
   dg,de=masked_mean(d,dv),masked_mean(d,dv,self.dynamic_steps-10)
  if arm!='dynamic_only':
   sv=batch['state_valid'].bool();s=self.state(batch['state'],sv)
   sg,se=masked_mean(s,sv),masked_mean(s,sv,self.state_steps-5)
  if arm=='dynamic_only':
   zeros=dg.new_zeros((len(dg),2*dg.shape[1]+64+26));features=torch.cat((dg,de,zeros),-1);available=dv.any(1)
  elif arm=='state_only':
   zeros=sg.new_zeros((len(sg),2*sg.shape[1]));features=torch.cat((zeros,sg,se,self.context(batch['context']),batch['current']),-1);available=sv.any(1)
  else:
   if arm!='no_eri':
    rd=self.dynamic_endpoint_from_state(de,s,dv[:,-10:].any(1),sv)
    rs=self.state_endpoint_from_dynamic(se,d,sv[:,-5:].any(1),dv);de=de+rd;se=se+rs
   features=torch.cat((dg,de,sg,se,self.context(batch['context']),batch['current']),-1);available=dv.any(1)|sv.any(1)
  logits=self.head(features)
  return torch.where(available[:,None],logits,self.prior_logits[None])
