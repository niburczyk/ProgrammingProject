import joblib

model = joblib.load('./model/svm_model_optimized.pkl')

print("n_support_:", model.n_support_)
print("support_ indices:", model.support_)
print("dual_coef_.shape:", model.dual_coef_.shape)
print("classes_:", model.classes_)
print("decision_function_shape:", model.decision_function_shape)

n_classes = len(model.classes_)
n_classifiers = model.dual_coef_.shape[0]
expected_ovo = n_classes * (n_classes - 1) // 2

if model.decision_function_shape == 'ovo':
    expected = n_classes * (n_classes - 1) // 2
elif model.decision_function_shape == 'ovr':
    expected = n_classes - 1
else:
    expected = None

if expected is not None and model.dual_coef_.shape[0] == expected:
    print(f"✅ Das Modell verwendet {model.decision_function_shape} mit erwarteter Klassifikatoranzahl {expected}.")
else:
    print("❌ Das Modell hat ein unübliches Format.")
