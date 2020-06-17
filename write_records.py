from os import path
from avsr.dataset_writer import RecordFileWriter
from avsr.io_utils import get_files


def main():
    r"""
    Writes audio, video, and labels .tfrecord files
    Please tailor this script for your own use case
    Required files:
    - train.scp and test.scp: must contain one example per line
        as a relative path from dataset root, e.g.

        foo/bar/file1noext
        foo/bar/file2noext
        bar/baz/file9000noext

    - label_file: must contain pairs of (example name - transcription)
        on each line, delimited by a space, e.g.

        foo/bar/file1noext if liberty is not entire it is not liberty
        foo/bar/file2noext you must unlearn what you have learned
        bar/baz/file9000noext i'm sorry i don't want to be an emperor

    - unit_list_file: defines the vocabulary, one token per line

    Before writing the video record files, it is required to process
    video clips in advance (with OpenFace) to store the aligned faces.
    Please refer to the provided example `extract_faces.py`.
    :return:
    """
    dataset_dir = '/path/to/your/dataset_root/'
    train_list = '/path/to/train.scp'
    test_list = '/path/to/test.scp'

    train = get_files(train_list, dataset_dir)
    test = get_files(test_list, dataset_dir)

    label_map = dict()
    for file in train+test:
        label_map[path.splitext(file)[0]] = path.splitext(file.split('dataset_name/')[-1])[0]

    writer = RecordFileWriter(
        train_files=train,
        test_files=test,
        label_map=label_map,
        )

    writer.write_labels_records(
        unit='character',
        unit_list_file='./avsr/misc/character_list',
        label_file='/path/to/label_file/',
        train_record_name='/output/path/characters_train.tfrecord',
        test_record_name='/output/path/characters_test.tfrecord',
    )

    writer.write_audio_records(
        content_type='feature',
        extension='wav',
        transform='logmel_stack_w8s3',
        snr_list=['clean', 10, 0, -5],
        target_sr=16000,
        noise_type='cafe',
        train_record_name='/output/path/logmel_train',
        test_record_name='/output/path/logmel_test',
    )

    writer.write_bmp_records(
        train_record_name='/output/path/rgb36lips_train.tfrecord',
        test_record_name='/output/path/rgb36lips_test.tfrecord',
        bmp_dir='/path/to/your/dataset_root/aligned_openface/',
        output_resolution=(36, 36),
        crop_lips=True,
    )


if __name__ == '__main__':
    main()

