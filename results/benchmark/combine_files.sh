mkdir -p $1/results
python ./tools/to_csv.py $1/cvx_bonmin/overview.json $1/results/cvx_bonmin.csv
python ./tools/to_csv.py $1/cvx_sbmiqp/overview.json $1/results/cvx_sbmiqp.csv
python ./tools/to_csv.py $1/cvx_sbmiqp_lb/overview.json $1/results/cvx_sbmiqp_lb.csv
python ./tools/to_csv.py $1/noncvx_bonmin/overview.json $1/results/noncvx_bonmin.csv
python ./tools/to_csv.py $1/noncvx_sbmiqp/overview.json $1/results/noncvx_sbmiqp.csv
python ./tools/to_csv.py $1/noncvx_sbmiqp_lb/overview.json $1/results/noncvx_sbmiqp_lb.csv

python ./tools/read_shot.py ./convex_set.csv $1/cvx_shot_mt $1/results/cvx_shot_mt.csv
python ./tools/read_shot.py ./convex_set.csv $1/cvx_shot_st $1/results/cvx_shot_st.csv
python ./tools/read_shot.py ./nonconvex_set.csv $1/noncvx_shot_mt $1/results/noncvx_shot_mt.csv
python ./tools/read_shot.py ./nonconvex_set.csv $1/noncvx_shot_st $1/results/noncvx_shot_st.csv

python ./tools/join_data.py $1/results/cvx.csv ./convex_set.csv $1/results/cvx_bonmin.csv $1/results/cvx_sbmiqp.csv $1/results/cvx_sbmiqp_lb.csv $1/results/cvx_shot_mt.csv $1/results/cvx_shot_st.csv
python ./tools/join_data.py $1/results/noncvx.csv ./nonconvex_set.csv $1/results/noncvx_bonmin.csv $1/results/noncvx_sbmiqp.csv $1/results/noncvx_sbmiqp_lb.csv $1/results/noncvx_shot_mt.csv $1/results/noncvx_shot_st.csv
