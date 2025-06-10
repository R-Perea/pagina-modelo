# my_datasets/basura/basura.py
import os
import tensorflow_datasets as tfds

class Basura(tfds.core.GeneratorBasedBuilder):
    """Dataset de clasificación de basura: glass, plastic, trash."""
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dataset de clasificación de basura: glass, plastic, trash.",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'label': tfds.features.ClassLabel(names=['glass', 'plastic', 'trash']),
            }),
            supervised_keys=('image', 'label'),
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.abspath('TrashType_Image_Dataset')
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'data_dir': data_dir},
            ),
        ]

    def _generate_examples(self, data_dir):
        """Genera pares (key, example) donde key es la ruta del archivo."""
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            normalized_label = label.strip().lower()
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(label_dir, fname)
                    yield path, {
                        'image': path,
                        'label': normalized_label,
                    }