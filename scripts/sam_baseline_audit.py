from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset


ROOT = Path(r"F:\NC-1-31_40\FILL")
START = 0
STOP = 800


for path in sorted(ROOT.glob("SAM_Mask_*_class.dat")):
    dataset = EnviDataset(path)
    classes = np.array(dataset.read_rows(START, STOP)[..., 0])
    names = dataset.info.metadata.get("class names", [])
    ids, counts = np.unique(classes, return_counts=True)
    print(f"\n{path.name}")
    for class_id, count in zip(ids, counts):
        class_id = int(class_id)
        name = names[class_id] if class_id < len(names) else str(class_id)
        print(f"  {class_id}: {name} = {int(count)}")
    dataset.close()


print("\nCorrected pilot versus selected ENVI SAM classifications")
PILOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_corrected_v2_0000_0800")
comparisons = {
    "carbonates": "SAM_Mask_Calcite_Dolomite_006_class.dat",
    "sulfates": "SAM_Mask_Anhydrite_Gypsum_006_class.dat",
    "clays": "SAM_Mask_Illite_Montmorillonite_Kaolinite_003_class.dat",
}
for group, reference_name in comparisons.items():
    predicted_dataset = EnviDataset(PILOT / group / "sam_only_classes.dat")
    reference_dataset = EnviDataset(ROOT / reference_name)
    predicted = np.asarray(predicted_dataset.read_rows(START, STOP)[..., 0])
    reference = np.asarray(reference_dataset.read_rows(START, STOP)[..., 0])
    agreement = float(np.mean(predicted == reference))
    differences = int(np.count_nonzero(predicted != reference))
    print(f"  {group}: agreement={agreement:.12f}, differences={differences}")
    predicted_dataset.close()
    reference_dataset.close()
