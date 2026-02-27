"""
    Bu dosya projede kullanılan makine öğrenmesi algoritmalarını içerir.
    Algoritmalar: 
    PegasosSVM - Primal Estimated sub-GrAdient SOlver for SVM algoritması
    MLP - Çok Katmanlı Algılayıcı (Multi-Layer Perceptron) ve Geri Yayılım (Backpropagation) algoritması
"""

import numpy as np
from scipy.sparse import issparse

"""
    Bu sınıf, Pegasos algoritması kullanan ve stokastik gradyan inişi temelli bir Lineer SVM sınıfıdır.
"""
class PegasosSVM:
    def __init__(self, learning_rate, lambda_param, epoch_number):
        self.learning_rate = learning_rate # Adım büyüklüğü (eta)
        self.lambda_param = lambda_param # Regülarizasyon katsayısı
        self.epoch_number = epoch_number # İterasyon sayısı
        self.w = None # Ağırlık vektörü
        self.b = 0  # Bias (sapma) değeri
        self.loss_history = [] # Hata geçmişini tutan liste

    """
         Bu fonksiyon, modeli verilen veri seti üzerinde eğitir.  
         min (lambda/2 * ||w||^2 + 1/m * sum(max(0, 1 - y_i * (w*x_i + b)))) 
    """

    def fit(self, X, y):
        
        y = y.ravel()
        
        # Sample ve feature sayısını almak için .shape fonksiyonu kullanıldı.
        sample_number, feature_number = X.shape 

        # Modelin her özelliği için bir ağırlık gerekli, .zeros fonksiyonu ile başta hepsi 0 olarak ayarlandı.
        self.w = np.zeros(feature_number)
        # Bias başlangıçta 0 olarak belirlendi.
        self.b = 0

        # Veri setinin seyrek olup olmaması kontrolü yapıldı.
        isSparse = issparse(X)

        for j in range(self.epoch_number):
            epoch_loss = 0 # Epoch hatasını tutacak değişken.
            # Stokastik gradyan inişi: Her bir örnek tek tek işleniyor.
            for i in range (sample_number): # Örnek satırlarına tek tek bakılıyor.

                if isSparse: 
                    # Sparse matris çarpımı optimizasyonu yapıldı.
                    score = X[i].dot(self.w)[0] + self.b
                else:
                    # Standart numpy çarpımı yapıldı.
                    score = np.dot(X[i], self.w) + self.b
        
                # Modelin hatası (hinge loss) kontrol ediliyor.

                # Eğer y * score < 1 ise, örnek marjinin içindedir veya yanlış sınıflandırılmıştır.
                result = y[i] * score # Gerçek cevap ile tahmin çarpıldı.

                if (result < 1): # Pegasos'a göre sonuç 1'den küçükse güncelleme yapılır
                    # Gradyan Güncellemesi: w = w - lr * (lambda*w - y*x)

                    if isSparse:
                        x_vector = X[i].toarray().ravel()
                        self.w -= self.learning_rate * (self.lambda_param * self.w - y[i]*x_vector)
                    else:
                        self.w -= self.learning_rate * (self.lambda_param * self.w - y[i]*X[i])
                    
                    # Bias güncellemesi yapıldı.
                    self.b -= self.learning_rate * (-y[i])
                    # Hata hesabı yapıldı.
                    epoch_loss += (1-result) # hinge loss hesaplaması
                else:
                    # Örnek doğru sınıflandırıldıysa ve marjin dışındaysa sadece regularization güncellemesi yapıldı.
                    self.w -= self.learning_rate * (self.lambda_param * self.w)   
            
            # Her epoch sonunda ortalama kayıp kaydedildi.
            mean_loss = epoch_loss/sample_number # ortalama kayıp hesaplandı
            self.loss_history.append(mean_loss) # bu değer kayıp listesine eklendi

    """
        Bu fonksiyon, eğitilen ağırlıkları kullanarak tahmin yapar.
        İşaret fonksiyonu sign(w*x + b) -> {-1, 1} döndürür.
    """
    def predict(self, X):

        if issparse(X):
            result = X.dot(self.w) + self.b
        else:
            result = np.dot(X, self.w) + self.b
        
        return np.sign(result)

"""
    Bu sınıf, dinamik katman yapısına sahip ileri beslemeli yapay sinir ağı sınıfıdır.
"""
class MLP:
    def __init__(self, layer_sizes, activation='relu', optimizer='adam', learning_rate=0.01, epochs=100, batch_size=32, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_sizes = layer_sizes # Katmanlardaki nöron sayıları
        self.activation_name = activation # Aktivasyon fonksiyonu 
        self.optimizer = optimizer # Optimizasyon algoritması 
        self.lr = learning_rate # Öğrenme hızı 
        self.epochs = epochs # Eğitim iterasyon sayısı 
        self.batch_size = batch_size # Paket boyutu
        self.momentum_param = momentum # Momentum katsayısı
        self.beta1 = beta1 # Adam algoritması için moment bozunma oranı
        self.beta2 = beta2 # Adam algoritması için karesel gradyan bozunma oranı
        self.epsilon = epsilon # Kararlılık sabiti
        
        self.weights = [] # Ağırlık matrislerinin listesi
        self.biases = [] # Bias vektörlerinin listesi
        self.loss_history = [] # Hata geçmişi
        
        # Geçmiş gradyanları tutmak için optimizer geçmişi
        self.v_w = [] # Momentum ve Adam için hız vektörleri 
        self.v_b = []
        self.s_w = [] # RMSProp ve Adam için karesel gradyan
        self.s_b = []
        self.t = 0    # Adam için zaman adımı
        
        self._initialize_weights() # Ağırlıklar için başlangıç değerleri atandı.


    """
        Bu fonksiyon ağırlık matrislerini seçilen aktivasyon fonksiyonuna uygun
        yöntemlerle rastgele başlatır ve optimizer değişkenlerini sıfırlar.
    """
    def _initialize_weights(self):
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            input_dim = self.layer_sizes[i] # Giriş katmanı nöron sayısı
            output_dim = self.layer_sizes[i+1] # Çıkış katmanı nöron sayısı
            
            # ReLU için He Initialization veya diğerleri için Xavier 
            if self.activation_name == 'relu':
                limit = np.sqrt(2 / input_dim)
            else:
                limit = np.sqrt(6 / (input_dim + output_dim))

            # Belirlenen limitler aralığında ağırlık matrisi oluşturuldu.  
            w = np.random.uniform(-limit, limit, (input_dim, output_dim))
            b = np.zeros((1, output_dim))
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Optimizeriçin momentum ve hız önbelleklerinin sıfırlanması
            self.v_w.append(np.zeros_like(w))
            self.v_b.append(np.zeros_like(b))
            self.s_w.append(np.zeros_like(w))
            self.s_b.append(np.zeros_like(b))


    """
        Bu fonksiyon, non-linear dönüşüm yaparak ağın karmaşık desenleri öğrenmesini sağlar.
    """
    def _activate(self, z):
        if self.activation_name == 'sigmoid': # Çıktıyı (0, 1) aralığına sıkıştırdı.
            return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        elif self.activation_name == 'tanh': # Çıktıyı (-1, 1) aralığına sıkıştırdı.
            return np.tanh(z)
        elif self.activation_name == 'relu': # Negatif değerleri sıfırladı, pozitifleri direkt geçirdi.
            return np.maximum(0, z)
        return z
    

    """
        Bu fonksiyon, geri yayılım sırasında zincir kuralını uygulamak için
        seçilen aktivasyon fonksiyonunun türevini hesaplar.
    """
    def _activate_derivative(self, a):
        if self.activation_name == 'sigmoid': # Sigmoid türevi: f'(x) = f(x) * (1 - f(x))
            return a * (1 - a)
        elif self.activation_name == 'tanh': # Tanh türevi: f'(x) = 1 - f(x)^2
            return 1 - a**2
        elif self.activation_name == 'relu': # ReLU türevi: x > 0 ise 1, değilse 0
            return (a > 0).astype(float)
        return 1


    """
        Bu fonksiyon, veriyi girişten çıkışa doğru katmanlardan geçirir.
        Her katmanın çıktısını (aktivasyon) geri yayılımda kullanmak üzere saklar.
    """
    def forward(self, X):
        activations = [X] # Önbellek
        inputs = [] # z değerleri (W*x + b) 
        input_data = X
        
        for i in range(len(self.weights)):
            # Lineer Hesaplama: z = W * x + b
            # Sparse matris kontrolü yapıldı.
            if issparse(input_data):
                z = input_data.dot(self.weights[i]) + self.biases[i]
            else:
                z = np.dot(input_data, self.weights[i]) + self.biases[i]
            
            inputs.append(z)
            
            # Aktivasyon fonksiyonu uygulandı.
            # Son katman her zaman Sigmoid (Binary Classification için)
            if i == len(self.weights) - 1:
                a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            else:
                a = self._activate(z)
            
            activations.append(a)
            input_data = a
            
        return activations


    """
        Bu fonksiyon, zincir kuralını kullanarak hatanın türevlerini hesaplar.
        Çıktıdan girişe doğru ilerler.
    """
    def backward(self, y, activations):
        m = y.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Son katman hatası (Cross Entropy türevi * Sigmoid türevi sadeleşmiş hali: dL/dz = (A - Y))
        A_last = activations[-1]
        delta = A_last - y.reshape(-1, 1)
        
        # Sondan başa doğru iterasyon yapıldı.
        for i in range(len(self.weights) - 1, -1, -1):
            A_prev = activations[i]
            
            # Ağırlık Gradyanı: dW = (A_prev.T * delta) / m
            # Sparse matris ise transpose işlemi farklıdır
            if issparse(A_prev):
                dw = A_prev.T.dot(delta) / m
            else:
                dw = np.dot(A_prev.T, delta) / m

            # Bias Gradyanı: db = sum(delta) / m    
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            # Hatayı bir önceki katmana taşıdı.
            # delta_new = (delta_old * W.T) * f'(z)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activate_derivative(activations[i])
                
        return gradients_w, gradients_b


    """
        Bu fonksiyon, geri yayılımdan gelen gradyanları kullanarak ağırlıkları günceller.
        Seçilen optimizasyon stratejisine göre matematiksel formülleri uygular.
    """
    def _update_parameters(self, grads_w, grads_b):
        self.t += 1 # Adam için zaman sayacı
        
        for i in range(len(self.weights)):
            # SGD (Stokastik Gradyan İnişi)
            # Ağırlıklar gradyanın tersi yönünde güncellendi.
            if self.optimizer == 'sgd':
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]
            
            # Momentum
            # Önceki güncellemelerin hızını koruyarak yerel minimumlara takılmayı engelledi.
            elif self.optimizer == 'momentum':
                self.v_w[i] = self.momentum_param * self.v_w[i] + self.lr * grads_w[i]
                self.v_b[i] = self.momentum_param * self.v_b[i] + self.lr * grads_b[i]
                self.weights[i] -= self.v_w[i]
                self.biases[i] -= self.v_b[i]
                
            # RMSProp (Root Mean Square Propagation)
            # Her parametre için öğrenme hızını adapte ederek büyük gradyanları sönümledi.
            elif self.optimizer == 'rmsprop':
                self.s_w[i] = self.beta1 * self.s_w[i] + (1 - self.beta1) * (grads_w[i]**2)
                self.s_b[i] = self.beta1 * self.s_b[i] + (1 - self.beta1) * (grads_b[i]**2)
                self.weights[i] -= self.lr * grads_w[i] / (np.sqrt(self.s_w[i]) + self.epsilon)
                self.biases[i] -= self.lr * grads_b[i] / (np.sqrt(self.s_b[i]) + self.epsilon)

            # Adam (Adaptive Moment Estimation)
            # Momentum ve RMSProp'un güçlü yönlerini birleştirip hem hızı hem yönü optimize etti.
            elif self.optimizer == 'adam':
                # Momentum (1. moment) Momentum - Hız tahmini
                self.v_w[i] = self.beta1 * self.v_w[i] + (1 - self.beta1) * grads_w[i]
                self.v_b[i] = self.beta1 * self.v_b[i] + (1 - self.beta1) * grads_b[i]
                
                # RMSProp (2. moment) RMSProp - Varyans tahmini
                self.s_w[i] = self.beta2 * self.s_w[i] + (1 - self.beta2) * (grads_w[i]**2)
                self.s_b[i] = self.beta2 * self.s_b[i] + (1 - self.beta2) * (grads_b[i]**2)
                
                # Başlangıçtaki sıfır sapmasını düzeltmek için Bias Correction yapıldı. 
                v_w_hat = self.v_w[i] / (1 - self.beta1**self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta1**self.t)
                s_w_hat = self.s_w[i] / (1 - self.beta2**self.t)
                s_b_hat = self.s_b[i] / (1 - self.beta2**self.t)
                
                # Son parametre güncellemesi yapıldı.
                self.weights[i] -= self.lr * v_w_hat / (np.sqrt(s_w_hat) + self.epsilon)
                self.biases[i] -= self.lr * v_b_hat / (np.sqrt(s_b_hat) + self.epsilon)

    """
        Bu fonksiyon, modeli Mini-Batch Gradient Descent kullanarak eğitir.
        Her epoch'ta veriyi karıştırır ve hatayı takip eder.
    """
    def fit(self, X, y):
        n_samples = X.shape[0]
        y = y.reshape(-1, 1) # Boyut tutarlılığı için.
        
        # Etiketler MLP için düzeltildi.
        if -1 in np.unique(y):
            y = np.where(y == -1, 0, 1)
        
        for epoch in range(self.epochs):
            # Shuffling yapıldı, bu modelin verilerin sırasını ezberlemesini önler.
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            # Sparse matris indexing hatasını önlemek için
            # (Sparse matrislerde direkt X[indices] yapmak bazen yavaştır ama Scipy destekler)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # İleri yayılım yapıldı.
                activations = self.forward(X_batch)
                
                # Geri yayılım yapıldı.
                gw, gb = self.backward(y_batch, activations)
                
                # 3. Parametre güncellemesi yapıldı.
                self._update_parameters(gw, gb)
            
            # Epoch sonu performans değerlendirmesi için loss hesaplandı.
            full_activations = self.forward(X)
            y_pred_prob = full_activations[-1]

            # Modelin tahmin ettiği olasılıkların gerçek değerden ne kadar saptığını ölçmek için Binary Cross Entropy hesabı yapıldı.
            loss = -np.mean(y * np.log(y_pred_prob + 1e-9) + (1 - y) * np.log(1 - y_pred_prob + 1e-9))
            self.loss_history.append(loss)
            
            # Her 10 epochta bir bilgi vermesi sağlandı.
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")

    """
        Bu fonksiyon, yeni veriler için sınıf tahmini yapar.
        Çıktı olasılığı 0.5'ten büyükse 1, değilse 0 sınıfına atar.
    """
    def predict(self, X):
        activations = self.forward(X)
        prob = activations[-1]
        return np.where(prob > 0.5, 1, 0) 