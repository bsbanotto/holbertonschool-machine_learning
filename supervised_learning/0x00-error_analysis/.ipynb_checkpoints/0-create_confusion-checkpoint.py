{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dce4ca33-64d7-4d21-b2d7-c73182672e04",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python3\n",
    "\"\"\"\n",
    "Function that creates a confusion matrix\n",
    "\"\"\"\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "def create_confusion_matrix(labels, logits):\n",
    "    \"\"\"\n",
    "    labels: one-hot numpy.ndarray of shape (m, classes) containing the correct labels\n",
    "        m: number of data points\n",
    "        classes: number of classes\n",
    "    logits: one-hot numpy.ndarray of shape (m, classes) containing the predicted labels\n",
    "    Returns a confusion numpy.ndarray of shape (classes, classes)\n",
    "        row indices: correct labels\n",
    "        column indixes: predicted labels\n",
    "    \"\"\"\n",
    "    confusion_matrix = np.zeros([labels.shape[1], logits.shape[1]])\n",
    "    # print(confusion_matrix)\n",
    "    true_label_index = np.where(labels == 1)[1]\n",
    "    true_logits_index = np.where(logits == 1)[1]\n",
    "    # print(true_label_index)\n",
    "    # print(true_logits_index)\n",
    "    indexes = list(zip(true_label_index, true_logits_index))\n",
    "    unique, counts = np.unique(indexes, return_counts=True, axis=0)\n",
    "    # print(unique)\n",
    "    # print(counts)\n",
    "    confusion_matrix[unique[:,0], unique[:,1]] = counts\n",
    "    # print(confusion_matrix)\n",
    "    \n",
    "    return(confusion_matrix)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
