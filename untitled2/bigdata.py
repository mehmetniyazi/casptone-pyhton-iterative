import numpy as np
from itertools import combinations,groupby
from collections import Counter
import pandas as pd
import sys
from IPython.display import display
import gc
import sqlite3




def getitemsidarray(orders):
    dtnames = orders.value_counts().to_frame()
    dtnames = dtnames.reset_index()
    dtnames = dtnames.rename(columns={'index': 'item_id', 'item_id': 'freq'})
    listofitim = dtnames['item_id'].unique()
    return  listofitim



def size(obj):
    return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))


def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else:
        return pd.Series(Counter(iterable)).rename("freq")
def order_count(order_item):
    return len(set(order_item.index))

def filterfive(item_stats,x):
    filtvar=item_stats.to_numpy()
    minsop=0.0004
    for i in filtvar:
        minsop=i[1]
        if int(i[0]<=x):
            minsop=i[1]
            return minsop
    return minsop


def get_item_pairs(order_item):
    gc.collect()
    order_item = order_item.reset_index().values
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]

        for item_pair in combinations(item_list, 2):
            yield item_pair
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','supportAB','freqA','supportA','freqB','supportB',
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))

    return rules[columns]


def association_rules(order_item, min_support,item_id):
    gc.collect()

    item_stats = freq(order_item).to_frame("freq")
    item_stats['support'] = item_stats['freq'] / order_count(order_item) * 100
    min_support=filterfive(item_stats,8)
    # Filter from order_item items below min support
    qualifying_items = item_stats[item_stats['support'] > min_support].index

    order_item = order_item[order_item.isin(qualifying_items)]
    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))
    # Filter from order_item orders with less than 2 items
    order_size = freq(order_item.index)
    qualifying_orders = order_size[order_size >= 2].index
    order_item = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))

    # Recalculate item frequency and support
    item_stats = freq(order_item).to_frame("freq")

    item_stats['support'] = item_stats['freq'] / order_count(order_item) * 100
    denframe=order_item.to_frame()
    denframe=denframe.reset_index()
    denframe=denframe.rename(columns={"index": "order id"})
    freqone = item_stats[item_stats['freq'] > 1]
    freqone=freqone.reset_index()
    freqone=freqone.rename(columns={"index": "item_id"})

    eeeeeeeee=pd.merge(denframe, freqone, left_on='item_id', right_on='item_id')
    ilk=eeeeeeeee['order_id'].unique()

    eeeeeeeee=eeeeeeeee.sort_values('order_id')

    eeeeeeeee=eeeeeeeee[eeeeeeeee['item_id']==item_id].sort_values('order_id')

    teeerer = eeeeeeeee['order_id'].unique()
    trter=np.setdiff1d(ilk,teeerer)

    order_item=order_item.drop(trter)

    if len(order_item)==0:
        print("Cok dusuk degere sahip olanı aramaya calıstın v bu seni olduruuuu")
        return order_item

    item_pair_gen = get_item_pairs(order_item)
    # Calculate item pair frequency and support
    item_pairs = freq(item_pair_gen).to_frame("freq")
    item_pairs['supportAB'] = item_pairs['freq'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))
    min_support = filterfive(item_stats, 5)
    # Filter from item_pairs those below min support
    item_pairs = item_pairs[item_pairs['supportAB'] >= min_support]
    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))

    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift'] = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    order_item=order_item.head(5)
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)



def loaddata(filename,skiprow,nrow,plus):
    gc.collect()
    orders = pd.read_csv(filename,skiprows=(skiprow+plus), nrows=nrow)
    orders=orders.head(1000000)


    if orders.columns[0]!='order_id':
        tem=orders.columns[0]
        temp=orders.columns[1]
        orders=orders.rename(columns={tem: 'order_id', temp: 'product_id'})

    print('orders -- dimensions: {0};   size: {1}'.format(orders.shape, size(orders)))

    orders = orders.set_index('order_id')['product_id'].rename('item_id')

    print('dimensions: {0};   size: {1};   unique_orders: {2};   unique_items: {3}'.format(orders.shape, size(orders),
                                                                                           len(orders.index.unique()),
                                                                                           len(orders.value_counts())))
    return orders





def loadnamesdata():
    item_name = pd.read_csv('products.csv')

    item_name = item_name.rename(columns={'product_id': 'item_id', 'product_name': 'item_name'})
    return item_name


def implementassoc(orders,minsupport,item_id)   :
    rules=association_rules(orders, 0.0000002120,item_id)
    return rules




def showresult(rules,item_name):
    columns = ['itemA', 'itemB', 'supportAB', 'freqA', 'supportA', 'freqB', 'supportB',
               'confidenceAtoB', 'confidenceBtoA', 'lift']

    try:
        rules = (rules
                 .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                 .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
        return rules[columns]
    except:
        return []



def iterfind(testi,skiprow,idd):
    gc.collect()
    test = testi
    plus = skiprow

    for i in range(10):
        gc.collect()
        try:
            orders=loaddata('order_products__prior.csv',(i*1000000),((i+1)*1000000),plus)
        except:
            break

        item_id=idd
        rules=implementassoc(orders,0.0000002120,item_id)


        item_name=loadnamesdata()
        rules_final=showresult(rules,item_name)
        if len(rules_final)==0:
            break

        tr=item_name.loc[item_name['item_id'] == idd]
        tr=tr['item_name'].unique()
        tr=tr.tolist()
        orders=orders.head(5)
        rules=rules.head(5)
        item_name=item_name.head(5)
        print(rules_final)
        rules_final = rules_final.loc[rules_final['itemA'] == tr[0]].sort_values('lift',ascending=False)
        print(rules_final)
        rules_final=rules_final.head(10).to_numpy()

        rules_final=rules_final.tolist()
        if len(test)<10:

            test.extend(rules_final)
            for i in test:
                print(i)

            test = sorted(test, key=lambda x: x[7], reverse=True)


        else:
            if(len(rules_final)!=0):
                for i in range(len(test)):
                    test[i][3]=rules_final[0][3]+test[i][3]

                for i in rules_final:
                    count=0
                    for z in test:

                        if i[1]==z[1]:
                            t=z
                            t[2]=(z[2]*z[3]+i[2]*i[3])/(z[3]+i[3])
                            t[4]=(z[4]+i[4])/2
                            t[5]=i[5]+t[5]
                            t[6] = (z[6] + i[6]) / 2
                            t[7]=(z[7]*z[3]+i[7]*z[3])/(z[3]+i[3])
                            t[8] = (z[8] + i[8]) / 2
                            t[8] = (z[8] + i[8]) / 2
                            test[count]=t
                        count += 1
            test = sorted(test, key=lambda x: x[9], reverse=True)
        for i in test:
            print(i)
    return test

def main():
    conn = sqlite3.connect('logdata.db')
    c = conn.cursor()
    c.execute("SELECT * FROM itemid")

    dd=list(c.fetchall())
    conn.commit()
    for i in range(len(dd)):
        dd[i]=list(dd[i])
        dd[i]=dd[i][1:]
    conn.close()

    dd.index([22825])
    for i in dd[0:5000]:
        print(i)
        aa=iterfind([],0,i[0])

        bb=iterfind(aa,10000000,i[0])
        gc.collect()
        cc=iterfind(bb,20000000,i[0])

        gc.collect()
        ee=iterfind(cc,30000000,i[0])

        gc.collect()
        for ii in ee:
            print(ii)
        conn = sqlite3.connect('logdata.db')

        for ii in ee:
            if ii[0]=="Organic D'Anjou Pears" or ii[1]=="Organic D'Anjou Pears":
                pass
            print(ii)
            c = conn.cursor()
            try:
                sql="INSERT INTO assocs(itemA,itemB,supportAB,freqA,supportA,freq,supportB,confidenceAtoB,confidenceBtoA,lift) Values('{0}','{1}',{2},{3},{4},{5},{6},{7},{8},{9})"\
                    .format(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7],ii[8],ii[9])
                print(sql)
                c.execute(sql)
            except:
                pass
            conn.commit()
        conn.close()

if __name__=="__main__":
    main()