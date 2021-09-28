
# Notes

- caution: the output folder will contain the results from the omparison AND the ref turbine.


# Run the optimization without fatigue constraint


Edit the script to turn off the fatigue, then
```
python driver.py
```


# Compare initial design to the mass optimization without the fatigue constraint

```
compare_designs --modeling_options modeling_options.yaml Madsen2019_10_forWEIS.yaml outputs_struct/blade_out.yaml
mv outputs outputs_withoutFatigue
```


# Run the optimization with fatigue constraint


Edit the script to turn on the fatigue, then
```
python driver.py
```


# Compare all results

```
compare_designs --modeling_options modeling_options.yaml Madsen2019_10_forWEIS.yaml outputs_struct/blade_out.yaml outputs_struct_
mv outputs outputs_withFatigue
```

