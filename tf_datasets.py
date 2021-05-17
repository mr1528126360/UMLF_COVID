import os
import shutil
from abc import ABC, abstractmethod
import random

import tensorflow as tf

import settings

import pandas as pd
import numpy as np
import cv2

class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
        self.raw_database_address = raw_database_address
        self.database_address = database_address

        #self.prepare_database()
        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()

        self.input_shape = self.get_input_shape()

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def prepare_database(self):
        pass

    @abstractmethod
    def get_train_val_test_folders(self):
        pass

    def check_number_of_samples_at_each_class_meet_minimum(self, folders, minimum):
        for folder in folders:
            if len(os.listdir(folder)) < 2 * minimum:
                raise Exception(f'There should be at least {2 * minimum} examples in each class. Class {folder} does not have that many examples')

    def _get_instances(self, k):
        def get_instances(class_dir_address):
            return tf.data.Dataset.list_files(class_dir_address, shuffle=True).take(2 * k)
        return get_instances

    def _get_parse_function(self):
        def parse_function(example_address):
            return example_address
        return parse_function

    def make_labels_dataset(self, n, k, meta_batch_size, steps_per_epoch, one_hot_labels):
        print('==========================================labels_dataset=====================================')
        labels_dataset = tf.data.Dataset.range(n)
        #print(labels_dataset.shape)
        if one_hot_labels:
            labels_dataset = labels_dataset.map(lambda example: tf.one_hot(example, depth=n))
        labels_dataset = labels_dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(2 * k),
            cycle_length=n,
            block_length=k
        )
        #print(labels_dataset)
        labels_dataset = labels_dataset.repeat(meta_batch_size)
        labels_dataset = labels_dataset.repeat(steps_per_epoch)
        count = 0
        return labels_dataset

    def get_supervised_meta_learning_dataset(
            self,
            folders,
            n,#n=5
            k,#k=1
            meta_batch_size,#meta_batch_size=1
            one_hot_labels=True,
            reshuffle_each_iteration=True,
    ):
        for class_name in folders:
            assert(len(os.listdir(class_name)) > 2 * k), f'The number of instances in each class should be larger ' \
                                                         f'than {2 * k}, however, the number of instances in' \
                                                         f' {class_name} are: {len(os.listdir(class_name))}'

        classes = [class_name + '*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size
        #print('folders:',folders)
        print('n:',n)
        print('len(classes):',classes)
        print('meta_batch_size:',meta_batch_size)
        print('steps_per_epoch:',steps_per_epoch)
        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        print('len dataset:',len(dataset))
        dataset = dataset.shuffle(buffer_size=len(folders), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.interleave(
            self._get_instances(k),
            cycle_length=n,
            block_length=k,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        #print(dataset)
        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        print(labels_dataset)
        dataset = dataset.batch(k, drop_remainder=False)
        print(dataset)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

    def get_umtra_dataset(
        self,
        folders,
        n,
        meta_batch_size,
        augmentation_function=None,
        one_hot_labels=True,
        reshuffle_each_iteration=True
    ):
        if augmentation_function is None:
            def same(x):
                return x

            augmentation_function = same

        def parse_umtra(example, label):
            return tf.stack((example, augmentation_function(example))), tf.stack((label, label))

        instances = list()
        for class_name in folders:
            instances.extend(os.path.join(class_name, file_name) for file_name in os.listdir(class_name))
        instances.sort()
        steps_per_epoch = len(instances) // n // meta_batch_size
        #读取train文件夹名
        #一共取len（instances）张图片
        #每次文件夹随机排列，随机从文件夹取k张图片
        len_folders = len(folders)
        len_instances = len(instances)
        k = 1
        step_num = len_instances//k//n
        pic_list = list()
        print('len_instaces:',len_instances)
        for num in range(step_num):
            random.shuffle(folders)
            #print('folders:',folders)
            for sub_folder in folders[:n]:
                fd = os.listdir(sub_folder)
                random.shuffle(fd)
                for sub in range(len(fd[:k])):
                    fd[sub] = sub_folder+fd[sub]
                    pic_list.append(fd[sub])
        #print(pic_list)
        #exit(0)
        print('len_pic:',len(pic_list))
        
        print('n:',n)
        print('meta_batch_size:',meta_batch_size)
        print('steps_per_epoch:',steps_per_epoch)
        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(pic_list)
        dataset = dataset.map(self._get_parse_function())

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(1, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.map(parse_umtra)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)
        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

#self.prepare_database()
#self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()
#self.input_shape = self.get_input_shape()
class OmniglotDatabase(Database):
    def __init__(
        self,
        random_seed,
        num_train_classes,
        num_val_classes,
    ):
        self.num_train_classes = num_train_classes
        self.num_val_classes = num_val_classes
        
        if str(settings.PROJECT_ROOT_ADDRESS)[-1] != '/':
            settings.PROJECT_ROOT_ADDRESS = settings.PROJECT_ROOT_ADDRESS+'/'
        if str(settings.OMNIGLOT_RAW_DATA_ADDRESS)[-1] != '/':
            settings.OMNIGLOT_RAW_DATA_ADDRESS = settings.OMNIGLOT_RAW_DATA_ADDRESS+'/' 
        print('PROJECT_ROOT_ADDRESS:',settings.PROJECT_ROOT_ADDRESS)
        print('OMNIGLOT_RAW_DATA_ADDRESS:',settings.OMNIGLOT_RAW_DATA_ADDRESS)
        

        super(OmniglotDatabase, self).__init__(
            settings.OMNIGLOT_RAW_DATA_ADDRESS,
            settings.PROJECT_ROOT_ADDRESS,
            random_seed=random_seed,
        )

    def get_input_shape(self):
        return 500, 500, 1#修改成500*500

    def get_train_val_test_folders(self):
        '''num_train_classes = self.num_train_classes
        num_val_classes = self.num_val_classes
        folders = [os.path.join(self.database_address, class_name) for class_name in os.listdir(self.database_address)]
        folders.sort()
        random.shuffle(folders)
        train_folders = folders[:num_train_classes]
        val_folders = folders[-num_val_classes:]
        test_folders = folders
        print(train_folders)
        print(type(train_folders))
        exit(0)
        return train_folders, val_folders, test_folders
    '''
        
        train_folders = ['bacteria/','COVID-19/','normal/','virus/']
        train_folders = [os.path.join(self.database_address, class_name) for class_name in train_folders]
        #val_folders = ['COVID-19/','normal_2/','Other pneumonia/']
        val_folders = ['COVID-19/','normal_2/','bacteria/','virus/']
        val_folders = [os.path.join(self.database_address, class_name) for class_name in val_folders]
        test_folders = ['test_ba/','test_COVID-19/','test_normal/','test_virus/']#
        test_folders = [os.path.join(self.database_address, class_name) for class_name in test_folders]

        return train_folders, val_folders, test_folders

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            print(tf.io.read_file)
            image = tf.image.resize(image, (500, 500))
            image = tf.cast(image, tf.float32)

            return 1 - (image / 255.)

        return parse_function


    def prepare_database(self):
        if os.path.exists(self.database_address):
            shutil.rmtree(self.database_address)
        os.mkdir(self.database_address)
        size_n=500
        k=0
        dir_list = ['bacteria/','COVID-19/','normal/','virus/','normal_2/','test_ba/','test_COVID-19/','test_normal/','test_virus/']
        for i in dir_list:
            destination_address = self.database_address+i
            if os.path.exists(destination_address):
                shutil.rmtree(destination_address)
            os.mkdir(destination_address)
            for filename in os.listdir(self.raw_database_address+i):
                if str(filename).find('.jpeg')>0 or str(filename).find('.jpg')>0 or str(filename).find('.png')>0:
                    character_address = self.raw_database_address+i+filename
                else:
                    continue
                try:
                    d = cv2.imread(character_address, 0)
                    d = cv2.resize(d, (size_n, size_n)) 
                    shutil.copy(character_address, destination_address)
                    print('character_address:',character_address)
                    k=k+1
                except:
                    pass
        
        '''data = np.array(pd.read_csv(self.raw_database_address+'cov.csv', index_col=False))
        for i in range(0, len(data)):
            b = self.raw_database_address + data[i][0] + '/' + data[i][1] + '/mod-rx/'
            try:
                for filename in os.listdir(b):
                    if filename.endswith('png'):  # listdir的参数是文件夹的路径
                        # print (filename) #此时的filename是文件夹中文件的名称
                        character_address = b + data[i][0] + '_' + data[i][1] + '_run-1_bp-chest_vp-pa_cr.png'
                        try:
                            d = cv2.imread(character_address, 0)
                            d = cv2.resize(d, (size_n, size_n))
                            destination_address=self.database_address+data[i][0] + '_' + data[i][1]+'/'
                            #print('destination_address:',destination_address)
                            if os.path.exists(destination_address):
                                shutil.rmtree(destination_address)
                            os.mkdir(destination_address)                                
                            shutil.copy(character_address, destination_address)
                            print('character_address:',character_address)
                            k=k+1
                        except:
                            pass
            except:
                pass'''
        print('number_of_images:',k)

        
class MiniImagenetDatabase(Database):
    def get_input_shape(self):
        return 84, 84, 3

    def __init__(self, random_seed=-1, config=None):
        super(MiniImagenetDatabase, self).__init__(
            settings.MINI_IMAGENET_RAW_DATA_ADDRESS,
            os.path.join(settings.PROJECT_ROOT_ADDRESS, 'data/mini-imagenet'),
            random_seed=random_seed,
        )

    def get_train_val_test_folders(self):
        dataset_folders = list()
        for dataset_type in ('train', 'val', 'test'):
            dataset_base_address = os.path.join(self.database_address, dataset_type)
            folders = [
                os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)
            ]
            dataset_folders.append(folders)
        return dataset_folders[0], dataset_folders[1], dataset_folders[2]

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.
        return parse_function

    def prepare_database(self):
        if not os.path.exists(self.database_address):
            shutil.copytree(self.raw_database_address, self.database_address)