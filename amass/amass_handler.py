from processor.processor import *


class AMASSDataProcessor:
    """
    AMASS datasets are in the form of axis-angles, converting them to rot matrix, then 
    back into exponential map results in the same vector
    Note: axis-angle is defined as a representation of rotation
          exponential map is a specific function that maps a 3D vector to a rotation (in some form)
    The link below also shows axis-angle and exponential maps are the same thing in AMASS
    https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/449664/Cetin_Doruk.pdf?sequence=1&isAllowed=y
    """

    DATA_KEY = 'poses'  # key to body pose data in AMASS npz dataset
    ROOT_ORIENTATION_INDEX = 3  # 0:3 in pose matrix are root orientation
    BODY_POSE_INDEX = 66  # 3:66 in pose matrix are body poses axis-angle representation (66: for hands)
    NUM_BODY_JOINTS = 21  # 21*3=63 joint values, +3 root orientation values, for a total of 66

    def __init__(self, root_dir):
        """
        AMASSDataProcessor class constructor that loads data from input root directory
        """
        self.root = root_dir
        self.train_set_path, self.test_set_path = [], []
        self.split_train_test()

    def __len__(self):
        """ Total number of motion types """
        return len(self.train_set_path) + len(self.test_set_path)

    def split_train_test(self):
        """
        Split train and test set based on sub directory structure, taking only one sample file from
        each sub directory as test set, with the remaining to be train set
        """
        parent_dir = [i for i in os.listdir(self.root) if '.' not in i]  # main categories
        for p in parent_dir:
            child_dir = os.listdir(self.root + p + '/')
            if len(child_dir) == 0:
                # Category has no data
                continue
            elif len(child_dir) == 1:
                # Put category data in train set only
                self.train_set_path.append(self.root + p + '/' + child_dir[0])
                continue
            else:
                # Leave 1 randomly selected dataset to test, rest to training
                self.test_set_path.append(
                    self.root + p + '/' + child_dir.pop(random.choice(range(len(child_dir)))))
                for c in child_dir:
                    if not c.startswith('.'):
                        self.train_set_path.append(self.root + p + '/' + c)

        return self.train_set_path, self.test_set_path

    def write_dataset_txt(self, output_dir):
        """
        Write train and test data in txt files and store under the following hierarchy
        (categories are defined by the directories under root, this structure follows
        the original structure in DMGNN):
        output_dir
            - train
                - category1
                    - category1_1.txt
                    - category1_2.txt
                    - category1_3.txt
                    ...
                - category2
                    - category2_1.txt
                    - category2_2.txt
                    - category2_3.txt
                    ...
                ...
            - test
                - category1
                    - category1_1.txt
                - category2
                    - category2_1.txt
                ...
        """

        def write_txt_helper(npz_file, txt_file):
            """ Write from .npz to txt """
            motion_data = np.load(npz_file)[self.DATA_KEY][:, :self.BODY_POSE_INDEX]
            np.savetxt(txt_file, motion_data, delimiter=',')
            print('File {} has been generated.'.format(txt_file))

        def batch_writing_helper(dataset_path, type='train'):
            """ Write all .npz data in dataset_path to output_dir/type/ as txt files """
            for t in dataset_path:
                # Generate dataset txt files
                parent_dir = t[len(self.root):].split('/')[0]
                full_path = output_dir + '{}/'.format(type) + parent_dir + '/'
                if not os.path.exists(full_path):
                    os.makedirs(full_path)

                index = len(os.listdir(full_path))
                write_txt_helper(t, full_path + parent_dir + '_{}.txt'.format(str(index)))

        batch_writing_helper(self.train_set_path, type='train')
        batch_writing_helper(self.test_set_path, type='test')


if __name__ == '__main__':
    accad = AMASSDataProcessor(root_dir='./ACCAD/')
    accad.write_dataset_txt(output_dir='../data/accad/')

    sfu = AMASSDataProcessor(root_dir='./SFU/')
    sfu.write_dataset_txt(output_dir='../data/sfu/')
