"""Figuras compartidas por la interfaz y el diagrama exportado."""
from matplotlib.figure import Figure
import numpy as np

def curves(history, selected_epoch=None):
    fig=Figure(figsize=(10,6), dpi=100, layout='constrained')
    axes=fig.subplots(2,2).flat
    for ax,key,title in zip(axes,('loss','accuracy','macro_f1','lr'),
                           ('Entropía cruzada sin ponderar','Acierto','F1 macro · igual peso por bioma','Learning rate utilizado')):
        available=[h for h in history if key in h]
        if available:
            ax.plot([h['epoch'] for h in available],[h[key] for h in available],
                    label='LR' if key=='lr' else 'Entrenamiento',color='#167d8d')
            val=[h for h in history if 'val_'+key in h]
            if val: ax.plot([h['epoch'] for h in val],[h['val_'+key] for h in val],label='Validación',color='#d18328')
            if selected_epoch is not None: ax.axvline(selected_epoch,color='#77818c',ls='--',alpha=.6,label='Modelo elegido')
            ax.legend(fontsize=8)
        elif history: ax.text(.5,.5,'No disponible en este modelo',ha='center',transform=ax.transAxes)
        ax.set(xlabel='Época',title=title); ax.grid(alpha=.18)
        if key in ('accuracy','macro_f1'):ax.set_ylim(0,1)
        if key=='lr' and available:ax.set_yscale('log')
    return fig

def confusion(metrics):
    names=metrics['classes']; matrix=np.array(metrics['confusion_matrix'])
    fig=Figure(figsize=(8,6),dpi=100,layout='constrained'); ax=fig.subplots()
    ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,
           xlabel='Bioma predicho',ylabel='Bioma real',title='Matriz de confusión · test reservado')
    ax.tick_params(axis='x',labelrotation=45)
    if len(names)<=20:
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',
                        color='white' if matrix[i,j]>matrix.max()/2 else '#14263d')
    return fig

def architecture(size=64, classes=6):
    fig=Figure(figsize=(10,8),dpi=130,facecolor='#f5f8fc')
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,10),ylim=(0,10)); ax.axis('off')
    ax.text(.5,9.5,'La red de biomas',fontsize=24,weight='bold',color='#162d48')
    ax.text(.5,9.05,f'CNN desde cero · entrada {size} × {size} · {classes} clases seleccionadas',fontsize=12)
    blocks=[('Imagen RGB',f'3 × {size} × {size}\nResize + normalización [−1, 1]'),
            ('Convolución 1 + ReLU + MaxPool',f'3 → 16 canales · kernel 5 × 5 · padding 2\nSalida: 16 × {size//2} × {size//2}'),
            ('Convolución 2 + ReLU + MaxPool',f'16 → 32 canales · kernel 5 × 5 · padding 2\nSalida: 32 × {size//4} × {size//4}'),
            ('Flatten + Linear + ReLU',f'{32*(size//4)**2} características → 120 activaciones'),
            ('Linear de salida',f'120 → {classes} logits (puntuaciones sin normalizar)')]
    for i,(title,subtitle) in enumerate(blocks):
        y=8.15-i*1.35
        ax.text(.65,y,title,fontsize=13,weight='bold',color='#135e70')
        ax.text(.65,y-.42,subtitle,fontsize=11,va='top',linespacing=1.5)
        if i<4: ax.annotate('',xy=(5,y-.95),xytext=(5,y-.65),arrowprops={'arrowstyle':'->','color':'#135e70'})
    ax.text(.65,1.25,'ENTRENAR',fontsize=11,weight='bold',color='#135e70')
    ax.text(.65,.65,'Logits + etiqueta real → CrossEntropyLoss\nBackpropagation + optimizador elegido',fontsize=11)
    ax.text(5.6,1.25,'CLASIFICAR UNA IMAGEN',fontsize=11,weight='bold',color='#135e70')
    ax.text(5.6,.65,'Logits → Softmax → porcentaje por bioma\nLos porcentajes suman 100 %',fontsize=11)
    return fig
