import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Désactiver les logs inutiles de TF pour gagner en vitesse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class AutonomousDrive(Node):
    def __init__(self):
        super().__init__('autonomous_drive')

        # 1. Créer la structure exacte (60x60 comme votre .npz)
        self.model = models.Sequential([
            layers.Input(shape=(60, 60, 3)), 
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2)
        ])

        # 2. CHARGEMENT SÉCURISÉ
        path = 'imita12_actor.pth'
        try:
            # On charge les poids uniquement pour ignorer la config incompatible
            self.model.load_weights(path)
            self.get_logger().info("Poids chargés avec succès !")
        except Exception:
            # Si load_weights échoue car c'est un fichier modèle complet, on utilise cette astuce :
            tmp_model = tf.keras.models.load_model(path, compile=False)
            self.model.set_weights(tmp_model.get_weights())
            self.get_logger().info("Poids extraits du modèle avec succès !")

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/fastbot_1/camera/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/fastbot_1/cmd_vel', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Utiliser 60x60 (votre dataset) 
            img = cv2.resize(cv_image, (60, 60))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            # Inférence
            prediction = self.model.predict(img, verbose=0)
            linear_x, angular_z = prediction[0]

            # 3. CORRECTION DE LA LENTEUR
            # Si le robot est trop lent, multipliez les sorties par un facteur (ex: 1.5 ou 2.0)
            twist = Twist()
            twist.linear.x = float(linear_x) * 1.5 
            twist.angular.z = float(angular_z) * 1.2
            
            self.cmd_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Erreur : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Correction de l'erreur de shutdown rclpy
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()