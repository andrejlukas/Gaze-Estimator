# Gaze-Estimator - Regresija smjera pogleda temeljena na analizi slika očiju 
*(Gaze Direction Regression based on Eye Image Analysis)*

Ovaj repozitorij sadrži izvorni kod razvijen u sklopu završnog rada na **Fakultetu elektrotehnike i računarstva (FER), Sveučilište u Zagrebu**. Cilj projekta bio je istražiti i implementirati sustav temeljen na metodama dubokog učenja za procjenu smjera pogleda iz slika očiju.

**Autor:** Andrej Lukas  
**Mentor:** akademik prof. dr. sc. Sven Lončarić  
**Datum:** Lipanj, 2025.  

---

## O projektu

Smjer pogleda osobe ključan je oblik neverbalne komunikacije koji nudi uvid u ljudsku kogniciju i namjere. Danas se detekcija pogleda primjenjuje u raznim područjima: analizi pažnje kupaca, prevenciji umora vozača i medicini. 

Ovaj projekt implementira metodu temeljenu na izgledu (*appearance-based method*) koristeći konvolucijske neuronske mreže kako bi odredio 3D vektor pogleda. Pristup direktno obrađuje izrezane fotografije lijevog i desnog oka, spaja ih i iz njih kroz mrežu procjenjuje dva kuta u sfernom koordinatnom sustavu:
- **Yaw (zakret)**: Rotacija oko vertikalne osi
- **Pitch (nagib)**: Rotacija oko horizontalne osi

## Arhitektura modela

Model se temelji na prilagođenoj **ResNet18** arhitekturi. Zbog asimetrije očiju (fotografije lijevog i desnog oka se mogu razlikovati), ulaz u mrežu sastoji se od slika oba oka.
- Slike lijevog i desnog oka spajaju se u jedan tenzor.
- Prvi konvolucijski sloj prilagođen je da na ulazu uzima **6 kanala** (3 kanala s lijevog i 3 kanala s desnog oka).
- Posljednji potpuno povezani (*fully connected*) sloj daje 2 izlaza koja predstavljaju kutove procjene (*yaw* i *pitch*).
- Funkcija pogreške koja se koristi je **L1 Loss** (Srednja apsolutna pogreška - MAE), a za optimizaciju se koristi prilagodljivi **Adam optimizer**.

## Skup podataka

Sustav se trenira i testira pomoću velikog skupa podataka za procjenu pogleda pod nazivom **[Gaze360](https://gaze360.csail.mit.edu/)**. 
Podaci se ucitavaju iz prilagođenih `train.txt`, `validation.txt` i `test.txt` datoteka koje sadrže putanje do slika te stvarne 3D komponente pogleda. Model primjenjuje jednostavan izračun transformacije 3D vektora pogleda (iz Kartezijevog sustava) u sferne vrijednosti (*yaw*, *pitch*). Tijekom eksperimenata su isprobane rezolucije slika: `16x16`, `32x32` i `64x64` pri čemu su veće rezolucije ostvarile preciznije rezultate na testnom skupu (između 9° i 12° prosječno).

Najmanja prosječna pogreška za ovaj skup podataka postignuta je za kombinaciju `64x64` rezolucije ulazne slike i `batch_size = 32` koja je iznosila oko 9.6° razlike između predviđenog smjera pogleda i točne vrijednosti.

[!slika1](https://github.com/andrejlukas/Gaze-Estimator/blob/8c02a9eafae79c730734fc3607ada72cb2ed23d0/images/Figure_1.png)

## Instalacija i pokretanje

pip install torch torchvision

pip install numpy matplotlib pillow tensorboard

python Gaze-estimation-final-code.py

python best-model-vizualization.py

tensorboard --logdir=runs_GAZE/


