## fusion module
First: You need to adjust the path of the res

### get the metrics of the res
Run  <code>bert_better.py</code> to fusion the results of the three modules.

Parameters:
1. <code>data_type</code>: select from <code>Lecard</code>, <code>ELAM</code>
2. <code>method</code>: select from <code>bert</code>, <code>bertpli</code>, <code>lawformer</code>, <code>shaobert</code>

one example:
<code>python bert_better.py --data_type Lecard --method bert</code>

You can also modify the flag and K parameters in each method to control the WRRF.

 
