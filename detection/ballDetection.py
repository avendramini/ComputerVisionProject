import cv2
import numpy as np
def ballDetection(frame):
    """
    Funzione per rilevare la palla in un frame e restituire una maschera binaria.
    
    Args:
        frame (any): Il frame da analizzare (immagine in formato BGR).
    
    Returns:
        mask (numpy.ndarray): Maschera binaria con 1 dove è presente la palla, 0 altrove.
    """
    # Converti il frame in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Applica un filtro gaussiano per ridurre il rumore
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Crea una maschera vuota
    mask = np.zeros_like(gray, dtype=np.uint8)
    
    # Rileva i cerchi usando HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=30)
    
    if circles is not None:
        # Se sono stati trovati cerchi, aggiungi il primo cerchio trovato alla maschera
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Disegna il cerchio sulla maschera
            cv2.circle(mask, (i[0], i[1]), i[2], 1, -1)
    
    return mask
