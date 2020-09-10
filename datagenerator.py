import keras
import cv2
from albumentations import Rotate
from random import randint

class DataGenerator3(keras.utils.Sequence):

    def __init__(self, data, batch_size=2, shuffle=True, target_size=(256, 256), name='train', aug=True):
        'Initialization'
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = -1
        self.path = 'dataset/'
        self.target_size = target_size
        self.flag_next = True
        self.index_pointer = 0
        self.name = name
        self.treshold_sum = 10
        self.num_of_items = 0
        self.aug = aug
        if self.aug:
            self.augmentation = Rotate(limit=180, p=0.5, border_mode=cv2.BORDER_CONSTANT)
        print(f'{self.name} generator initialization ...')
        self.generate_flags()
        self.__calculate_epoch_size__()
        self.on_epoch_end()

    def __calculate_epoch_size__(self):

        self.epoch_size = 0
        item_counter = 0
        for item in self.data:
            item_counter += 1
            counter = 0
            mask = np.load(self.path + item + '_mask_segment.npy')
            for i in range(mask.shape[2] - 2):
                mask_slice = np.sum(mask[:, :, i])
                if mask_slice > self.treshold_sum:
                    counter += 1
            if counter > 0:
                num_of_batches = int(np.ceil(counter / self.batch_size))
                self.epoch_size += num_of_batches
            # print('item: ', item, 'num of slices', counter, 'num of batches', num_of_batches)
        number_of_pats_without_masks = len(self.flags) / self.batch_size
        # self.epoch_size += int(number_of_pats_without_masks)
        print('done!')
        self.num_of_items = item_counter
        print(f'dataset size: {item_counter} items')
        print(f'estimated epoch length: {self.epoch_size} batches')

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.on_epoch_end()
        return self.epoch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        if self.flag_next:
            self.counter += 1
            self.x_chunk, self.y_chunk = self.__data_generation(self.data[self.counter])
            self.indexes = np.arange(self.x_chunk.shape[0])
            self.index_pointer = 0
        indexes = self.indexes[self.index_pointer * self.batch_size:(self.index_pointer + 1) * self.batch_size]
        self.index_pointer += 1

        self.flag_next = False

        if (self.indexes[-1] + 1) <= (self.index_pointer) * self.batch_size:
            self.flag_next = True
        return self.x_chunk[indexes, :, :, :], self.y_chunk[indexes, :, :]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.counter = -1
        self.flag_next = True
        if self.shuffle == True:
            np.random.shuffle(self.data)

    def __data_generation(self, item_ID):
        'Generates data containing batch_size samples'
        img = np.load(self.path + item_ID + '_clean.npy')
        mask = np.load(self.path + item_ID + '_mask_segment.npy')
        X, y = self.create_batch(img, mask)
        return X, y

    def resize(self, image, is_mask=False):
        size = self.target_size[0]
        x_w = image.shape[1]
        x_h = image.shape[0]
        x_scale = size / x_w
        y_scale = size / x_h
        # INTER_AREA
        # INTER_LANCZOS4
        resized_img = cv2.resize(image, (0, 0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LANCZOS4)
        # resized_img = image
        if is_mask:
            # resized_img[resized_img > 0.1] = 1
            # resized_img[resized_img < 0.1] = 0
            pass
        return resized_img

    def create_batch(self, img, mask_):

        self.generate_flags()

        img = img[0, :, :, :]
        num_of_slices = img.shape[2]
        mask = mask_
        X = []
        Y = []

        counter = 1
        while counter < (num_of_slices - 2):

            current_sum = np.sum(mask[:, :, counter])
            # add images with mask
            if current_sum > self.treshold_sum:
                if counter in self.flags:
                    self.correct_flags(counter)
                data = {}
                resized_mask = self.resize(mask[:, :, counter], is_mask=True)
                data["mask"] = resized_mask

                x_local = img[:, :, counter - 1:counter + 2]
                x_local = self.resize(x_local)
                data["image"] = x_local

                if self.name == 'train' and self.aug:
                    # flips
                    data_augmented = self.augmentation(**data)
                    x_local = data_augmented["image"]
                    resized_mask = data_augmented["mask"]

                y_local = np.expand_dims(resized_mask, axis=-1)
                Y.append(y_local)
                X.append(self.norm(x_local))
                counter += 1
            else:
                # add images without mask

                #                 if counter in self.flags and self.flags[counter] == False:
                #                     resized_mask = self.resize(mask[:,:, counter], is_mask=True)
                #                     y_local = np.expand_dims(resized_mask, axis=-1)
                #                     x_local = img[:, :, counter-1:counter+2]
                #                     x_local = self.resize(x_local)
                #                     Y.append(y_local)
                #                     X.append(self.norm(x_local))
                #                     self.flags[counter] = True

                counter += 1
                continue

        patch_X = np.stack(X, axis=0)
        patch_Y = np.stack(Y, axis=0)
        return patch_X, patch_Y

    def norm(self, image):
        image = np.squeeze(image)
        normalize = lambda x: x / np.max(x) - 0.5
        normalized = normalize(image)
        return normalized

    def get_epoch_size(self):
        return self.epoch_size

    def generate_intervals(self, items, bias):
        result = []
        for item in items:
            result.append(randint(item - bias, item + bias))
        return result

    def make_flag_dict(self, intervals):
        result = []
        for inter in intervals:
            result.extend(list(range(inter, inter + self.batch_size)))
        statuses = [False] * len(result)
        return dict(zip(result, statuses))

    def generate_flags(self):
        init_items = [150]
        intervals = self.generate_intervals(init_items, 20)
        self.flags = self.make_flag_dict(intervals)

    def correct_flags(self, counter):
        self.flags.pop(counter, None)
        self.flags[counter + self.batch_size] = False

