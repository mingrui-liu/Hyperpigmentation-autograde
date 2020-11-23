import train_fsl







def plot_acc(history):
  
    plt.title('Training and validation root_mean_squared_error')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def plo_loss(history):

    plt.title('Training and validation loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()


def plot_rmse(history):
  
    plt.title('Training and validation RMSE')
    plt.plot(history.history['root_mean_squared_error'], label='RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label = 'val_RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='lower right')
    plt.show()

def confusion_matrix(test_labels,prediction):
    classes=[0,1,2,3,4,5,6]
    con_mat  = tf.math.confusion_matrix(
        test_labels, prediction, num_classes=None, weights=None, dtype=tf.dtypes.int32,
        name=None
    ).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                         index = classes, 
                         columns = classes)


    import seaborn as sns
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



history,evaluate,prediction,test_labels = train_fsl.train_model()

plot_loss(history)
plot_rmse(history)
plot_acc(history)
confusion_matrix(test_labels,prediction)


#print("Loss: {:1.2}".format(loss))
#print("RMSE: {:1.2}".format (RMSE))
#print("Accuracy: {:2.2%}".format(accuracy))
    


