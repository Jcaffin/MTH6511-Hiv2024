using BundleAdjustmentModels

df = problems_df()
filter_df = df[ ( df.nvar .≤ 34000 ), :]

name = filter_df[1, :name]

model = BundleAdjustmentModel(name)