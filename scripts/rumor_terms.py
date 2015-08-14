import re

rumor_terms = {
    ### DC power outage Rumors ##
    'explosion':{
        '$or':[
            {'text':re.compile('explosion',re.IGNORECASE)},
            {'text':re.compile('blast',re.IGNORECASE)},
            {'text':re.compile('boom',re.IGNORECASE)}
        ]
    },
    'foul_play':{
        '$or':[
            {'text':re.compile('foul',re.IGNORECASE)},
            {'text':re.compile('terror',re.IGNORECASE)},
            {'text':re.compile('attack',re.IGNORECASE)},
            {'text':re.compile('hack',re.IGNORECASE)}
        ]
    },
    ### WestJet Rumors ##
    'signal':{
        '$or':[
            {'text':re.compile('squawk',re.IGNORECASE)},
            {'text':re.compile('code',re.IGNORECASE)},
            {'text':re.compile('alarm',re.IGNORECASE)},
            {'text':re.compile('signal',re.IGNORECASE)},
            {'text':re.compile('button',re.IGNORECASE)},
            {'text':re.compile('transponder',re.IGNORECASE)},
            {'text':re.compile('7500')}
        ]
    },
    'hijacking':{
        '$or':[
            {'text':re.compile('squawk',re.IGNORECASE)},
            {'text':re.compile('signal',re.IGNORECASE)},
            {'text':re.compile('button',re.IGNORECASE)},
            {'text':re.compile('plane',re.IGNORECASE)},
            {'text':re.compile('pilot',re.IGNORECASE)},
            {'text':re.compile('transponder',re.IGNORECASE)},
            {'text':re.compile('west',re.IGNORECASE)},
            {'text':re.compile('flight',re.IGNORECASE)},
            {'text':re.compile('jet',re.IGNORECASE)},
            {'text':re.compile('7500')},
        ]
    },
   ### MH17 RUMORS ##
    'blackbox':{
        '$and':[
            {
                '$or':[
                    {'text':re.compile('rebels',re.IGNORECASE)},
                    {'text':re.compile('separatists',re.IGNORECASE)},
                    {'text':re.compile('terrorists',re.IGNORECASE)}
                ]
            },
            {
                '$or':[
                    {'text':re.compile('blackbox',re.IGNORECASE)},
                    {'text':re.compile('black box',re.IGNORECASE)},
                    {'text':re.compile('recorder',re.IGNORECASE)}
                ]
            }
        ]
    },
	# EBOLA RUMORS ##
    'red_cross':{
        '$and':[
            {'text':re.compile('vaccine',re.IGNORECASE)},
            {
                '$or':[
                    {'text':re.compile('red cross|redcross',re.IGNORECASE)},
                    {'text':re.compile(' ARC ',re.IGNORECASE)},
                    {'text':re.compile(' IRC ',re.IGNORECASE)}
                ]
            }
        ]
    },
    'semen':{
        '$or':[
            {'text':re.compile('semen',re.IGNORECASE)},
            {'text':re.compile('breast milk',re.IGNORECASE)},
            {'text':re.compile('breastmilk,re.IGNORECASE')}
        ]
    },
    'frontier':{
        '$and':[
            {'text':re.compile('frontier',re.IGNORECASE)},
            {
                '$or':[
                    {'text':re.compile('ban',re.IGNORECASE)},
                    {'text':re.compile('grounded',re.IGNORECASE)}
                ]
            }
        ]
    },
    'quarantine':
    {
        '$or':[
            {'text':re.compile('camps',re.IGNORECASE)},
            {'text':re.compile('trailers',re.IGNORECASE)}
        ]
    },
    ## SYDNEY SIEGE RUMORS ##
    'gunmen':
    {
        '$or':[
            {
                '$and':[
                    {'text':re.compile('gunman',re.IGNORECASE)},
                    {
                        '$or':[
                            {'text':re.compile('single',re.IGNORECASE)},
                            {'text':re.compile('one',re.IGNORECASE)},
                            {'text':re.compile(' 1 ',re.IGNORECASE)},
                            {'text':re.compile('only',re.IGNORECASE)},
                        ]
                    }
                ]
            },
            {
                '$and':[
                    {'text':re.compile('gunmen',re.IGNORECASE)},
                    {
                        '$or':[
                            {'text':re.compile(' 2',re.IGNORECASE)},
                            {'text':re.compile('2 ',re.IGNORECASE)},
                            {'text':re.compile(' 3',re.IGNORECASE)},
                            {'text':re.compile('3 ',re.IGNORECASE)},
                            {'text':re.compile('two',re.IGNORECASE)},
                            {'text':re.compile('three',re.IGNORECASE)},
                            {'text':re.compile('multiple',re.IGNORECASE)},
                        ]
                    }
                ]
            }
        ]
    },
    # unused
    'isis':
    {
        '$or':[
            {'text':re.compile(' isis',re.IGNORECASE)},
            {'text':re.compile('#isis',re.IGNORECASE)},
        ]
    },
    'suicide':
    {
        '$or':[
            {'text':re.compile('suicide',re.IGNORECASE)},
            {'text':re.compile(' belt',re.IGNORECASE)},
            {'text':re.compile(' vest',re.IGNORECASE)},
            {'text':re.compile('backpack',re.IGNORECASE)},
        ]
    },
    'airspace':
    {
        '$or':[
            {'text':re.compile('airspace',re.IGNORECASE)},
            {'text':re.compile('air space',re.IGNORECASE)},
            {'text':re.compile('flights',re.IGNORECASE)},
            {'text':re.compile('no-fly',re.IGNORECASE)},
            {'text':re.compile('no fly',re.IGNORECASE)}
        ]
    },
    # unused
    'tweet':
    {
        '$and':[
            {'text':re.compile('police',re.IGNORECASE)},
            {
                '$or':[
                    {'text':re.compile('ask',re.IGNORECASE)},
                    {'text':re.compile('request',re.IGNORECASE)}
                ]
            },
            {
                '$or':[
                    {'text':re.compile('tweet',re.IGNORECASE)},
                    {'text':re.compile('post',re.IGNORECASE)},
                    {'text':re.compile('social media',re.IGNORECASE)},
                    {'text':re.compile('share',re.IGNORECASE)}
                ]
            }
        ]
    },
    # unused
    'priest':
    {
        'text':re.compile('priest',re.IGNORECASE)
    },
    'hadley':
    {
        '$and':[
            {'text':re.compile('hostage',re.IGNORECASE)},
            {
                '$or':[
                    {'text':re.compile('hadley',re.IGNORECASE)},
                    {'text':re.compile('radio host',re.IGNORECASE)}
                ]
            }
        ]
    },
    'lakemba':
    {
        '$and':[
            {'text':re.compile('lakemba',re.IGNORECASE)},
            {'text':re.compile('^((?!vigil).)*$',re.IGNORECASE)},
        ]
    },
    'flag':
    {
        '$and':[
            {'text':re.compile('flag',re.IGNORECASE)},
            {
                '$or':[
                    {'text':re.compile(' isis',re.IGNORECASE)},
                    {'text':re.compile('#isis',re.IGNORECASE)},
                    {'text':re.compile('isil',re.IGNORECASE)}
                ]
            }
        ]
    },
    'americans_onboard':
    {
        '$and':[
            {
                '$or':[
                    {'text':re.compile('passenger',re.IGNORECASE)},
                    {'text':re.compile('board',re.IGNORECASE)},
                    {'text':re.compile('23',re.IGNORECASE)},
                ]
            },
            {
                '$or':[
                    {'text':re.compile('americans',re.IGNORECASE)},
                    {'text':re.compile('us citizen',re.IGNORECASE)},
                ]
            }
        ]
    },
    'rebels':
    {
        '$and':[
            {'text':re.compile('ukrain',re.IGNORECASE)},
            {'text':re.compile('shot',re.IGNORECASE)},
            {
                '$or':[
                    {'text':re.compile(' rebel',re.IGNORECASE)},
                    {'text':re.compile('separatist',re.IGNORECASE)}
                ]
            }
        ]
    },
    'american_falseflag':
    {
        '$and':[
            {
                '$or':[
                    {'text':re.compile('falseflag',re.IGNORECASE)},
                    {'text':re.compile('false flag',re.IGNORECASE)},
                ]
            },
            {
                '$or':[
                    {'text':re.compile('america',re.IGNORECASE)},
                    {'text':re.compile('usa',re.IGNORECASE)},
                ]
            }
        ]
    },
    'israel_falseflag':
    {
        '$and':[
            {
                '$or':[
                    {'text':re.compile('falseflag',re.IGNORECASE)},
                    {'text':re.compile('false flag',re.IGNORECASE)},
                ]
            },
            {
                '$or':[
                    {'text':re.compile('israel',re.IGNORECASE)},
                    {'text':re.compile('zion',re.IGNORECASE)},
                ]
            }
        ]
    },
    'same_plane':
    {
        '$and':[
            {'text':re.compile('mh17',re.IGNORECASE)},
            {'text':re.compile('mh370',re.IGNORECASE)},
            {'text':re.compile('same',re.IGNORECASE)},
        ]
    },
    'blackbox':{
        '$and':[
            {
                '$or':[
                    {'text':re.compile('rebels',re.IGNORECASE)},
                    {'text':re.compile('separatists',re.IGNORECASE)},
                    {'text':re.compile('terrorists',re.IGNORECASE)}
                ]
            },
            {
                '$or':[
                    {'text':re.compile('blackbox',re.IGNORECASE)},
                    {'text':re.compile('black box',re.IGNORECASE)},
                    {'text':re.compile('recorder',re.IGNORECASE)}
                ]
            }
        ]
    },
}

event_rumor_map = {
    'sydneysiege':['suicide','flag','lakemba','airspace','hadley'],#fix this
    'mh17':['americans_onboard'],
    'westjet':['hijacking'],
    'dc_power_outage':['foulplay'],
    #'donetsk':['nuclear_detonation'],
    'baltimore':['purse','church_fire',]
}

compression_rumor_map = {
    'sydneysiege_cache':['suicide','flag','lakemba','airspace','hadley'],#fix this
    'rumor_compression':['americans_onboard','hijacking','foul_play','purse','church_fire',]
    #'donetsk':['nuclear_detonation'],
}

filter_words = {
    'hadley':['hostage','hadley','radio','host'],
    'lakemba':['lakemba'],
    'flag':['flag','isis','isil'],
    'suicide':['suicide','belt','vest','backpack'],
    'airspace':['airspace','air','space','flights','no','fly','no-fly'],
    'americans_onboard':['passenger','board','23','americans','us','citizen'],
    'hijacking':['squawk','signal','button','plane','pilot','transponder','west','flight','jet','7500'],
    'purse':[],
    'church_fire':[]
}

event_terms = {
    'sydneysiege':['sydneysiege','martinplacesiege','haron','monis','haronmonis','illridewithyou','martinplace','sydney','chocolate shop','nswpolice','prime minister','tony abbott','witness','lindt','siege','hostage','hostages','martin place','terrorise','terrorize','terrorists','flag'],
    'mh17':['sa11','sa-11','ukraine','malaysia','airlines','mh17','kiev','mh017','torez','donetsk','airline','plane','buk'],
    'WestJet_Hijacking':[],
    'baltimore':[],
    'dc_power_outage':[]
}
