"""Known-logit CPU fixture for the new complete-set inference path."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new_complete_set_eval',D/'full_eval.py')
evaluate=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluate)
from experiments.permworld_encoding.core import encode


class KnownLogits(torch.nn.Module):
    def forward(self,ids,mask):
        assert not self.training and not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled('cpu')
        output=torch.zeros(*ids.shape,188,dtype=torch.float32)
        output[:,:,evaluate.TOKEN_TO_ID['01']]=2.
        for row,n in enumerate(ids[:,2].tolist()):
            output[row,2*n+6,evaluate.TOKEN_TO_ID['<EOS>']]=3.
        return output


def main():
    lengths=np.array([10,10,11,11,12,12])
    inputs=np.full((6,1,64),evaluate.TOKEN_TO_ID['<PAD>'],dtype=np.uint16)
    for i,n in enumerate(lengths):
        prefix=encode(tuple(range(1,int(n)+1)),'one_line')
        inputs[i,0,:len(prefix)]=prefix
    labels=np.zeros((6,1,len(evaluate.TASKS)),dtype=np.uint8)
    labels[:,0,evaluate.TASKS.index('descents')]=[1,0,0,1,1,0]
    data={f'{s}_{name}':value for s in ('train','validation') for name,value in
          [('lengths',lengths),('labels',labels),('one_line_input',inputs)]}
    protocol=dict(lengths=[10,11,12],samples_per_length={'train':2,'validation':2},
        eos_token_id=evaluate.TOKEN_TO_ID['<EOS>'],
        validation_reproduction_tolerances=dict(accuracy_absolute=0.,ce_absolute=1e-5,ce_relative=1e-6))
    net=KnownLogits().eval(); job=dict(run_id='toy',architecture='toy',encoding='one_line',task='descents',stage_path='toy-only')
    all_arrays={}
    for split in ('train','validation'):
        arrays=evaluate.evaluate_split(net,data,job,split,torch.device('cpu'),protocol)
        rows,summary=evaluate.summarize_arrays(arrays,split,protocol)
        assert np.array_equal(arrays[split+'_sample_index'],np.arange(6))
        assert summary['answer_accuracy']==.5 and summary['eos_accuracy']==1.
        assert abs(summary['answer_ce']-(math.log(math.exp(2)+187)-1))<1e-6
        assert abs(summary['eos_ce']-(math.log(math.exp(2)+math.exp(3)+186)-3))<1e-6
        assert summary['combined_ce']==.5*(summary['answer_ce']+summary['eos_ce'])
        all_arrays.update(arrays)
    reference=dict(validation={'per_length':rows,**summary})
    differences=evaluate.reproduce(job,reference,rows,summary,protocol)
    assert len(differences)==20 and all(r['passed'] and r['signed_difference']==0 for r in differences)
    # Check wrong mode/split rejection before any forward.
    for bad in ('test',):
        try:evaluate.evaluate_split(net,data,job,bad,torch.device('cpu'),protocol)
        except AssertionError:pass
        else:raise AssertionError('Reserved test accepted')
    net.train()
    try:evaluate.evaluate_split(net,data,job,'train',torch.device('cpu'),protocol)
    except AssertionError:pass
    else:raise AssertionError('Training mode accepted')
    evaluate.json_write(D/'evaluation_fixture.json',dict(status='passed',CPU_only=True,
        known_logit_accuracy_and_CE=True,answer_and_EOS_positions_checked=True,
        complete_sample_coverage=True,complete_train_and_validation=True,
        combined_loss_identity=True,validation_reproduction_identity=True,
        training_mode_rejected=True,reserved_test_rejected=True,
        model_forward_in_inference_mode=True,source_models_loaded=False,
        full_eval_sha256=evaluate.sha(D/'full_eval.py')))
    print('Known-logit complete-set CPU fixture: passed')


if __name__=='__main__':main()
